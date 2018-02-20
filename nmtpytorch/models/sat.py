# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.image_encoder import ImageEncoder
from ..vocabulary import Vocabulary
from ..utils.misc import pbar
from ..utils.data import to_var
from ..utils.nn import set_learnable
from ..utils.topology import Topology
from ..layers import FF

from ..datasets import Multi30kDataset


# Here we use an attention mechanism which may be
# different for general purpose attention defined in
# layers/attention since we're trying to replicate the exact
# architecture of the paper.


class Attention(nn.Module):
    """Attention layer for attention."""
    def __init__(self, ctx_dim, hid_dim):
        super().__init__()

        self.aux_loss = None

        # Visual context
        self.ctx_dim = ctx_dim

        # Hidden state of the RNN (or another arbitrary entity)
        self.hid_dim = hid_dim

        # w_att, b_att -> Final reduction transformation
        # to obtain attention scores
        self.att = nn.Linear(self.ctx_dim, 1)

        # Adaptor from RNN's hidden dim to visual context dim
        self.hid2ctx = nn.Linear(self.hid_dim, self.ctx_dim, bias=False)

        # Additional context projection within same dimensionality
        self.ctx2ctx = nn.Linear(self.ctx_dim, self.ctx_dim)

        self.reset_parameters()

    def reset_parameters(self):
        # Reinitialize with normal(stdev=0.01)
        nn.init.normal(self.att.weight, std=0.01)
        nn.init.normal(self.hid2ctx.weight, std=0.01)
        nn.init.normal(self.ctx2ctx.weight, std=0.01)

        # Set biases to zero
        self.att.bias.data.zero_()
        self.ctx2ctx.bias.data.zero_()

    def forward(self, ctx, hid):
        # Fuse hidden state of decoder and image context
        # unsqueeze to add a singleton middle dimension for bcast
        fusion = F.tanh(self.hid2ctx(hid).unsqueeze(1) + self.ctx2ctx(ctx))

        # Normalize to have attention scores
        alpha_t = F.softmax(self.att(fusion), dim=1)

        # Compute attended context
        z_t = torch.sum(alpha_t * ctx, dim=1)

        return alpha_t, z_t


class ShowAttendAndTell(nn.Module):
    r"""An Implementation of 'Show, attend and tell' image captioning paper.

    Paper: http://www.jmlr.org/proceedings/papers/v37/xuc15.pdf
    Reference implementation: https://github.com/kelvinxu/arctic-captions

    Arguments:
        vocab (Vocabulary): a Vocabulary instance for word mappings
        cnn_type (str, optional): CNN model to be used for image encoder
        cnn_layer (str, optional): CNN layer to extract the features
        cnn_trainable (bool, optional): Should CNN weights be trained along
        cnn_pretrained (bool, optional): Should CNN weights be pre-trained
        emb_dim (int, optional): Word embedding dimensionality
        dec_dim (int, optional): LSTM dimensionality for decoder
        dropout (float, optional): Dropout rate for dropouts in the network
        prev2out (bool, optional): Use previous embeddings in the final term
        ctx2out (bool, optional): Use attended context in the final term
        selector (bool, optional): Use a learned scalar to suppress context
        alpha_c (float, optional): Doubly stochastic regularization

    """

    def __init__(self, opts, logger=None):
        """This should only be used for argument processing."""
        super().__init__()

        self.vocabs = {}
        self.datasets = {}
        self.logger = logger
        self.print = print if self.logger is None else self.logger.info

        # Get a copy of model options coming from config
        # Consume them with .pop(), integrating model defaults
        # What is left inside kwargs -> Unused arguments
        kwargs = opts.model.copy()

        # Get direction and parse it
        opts.model['direction'] = kwargs.pop('direction', 'image->en')
        self.topology = Topology(opts.model['direction'])

        assert len(self.topology.get_src_langs()) == 0, \
            "Source languages not supported for this model."
        assert len(self.topology.get_trg_langs()) == 1, \
            "Multiple target languages not supported."

        self.tl = self.topology.get_trg_langs()[0]

        # Load vocabularies here
        self.vocabs[self.tl] = Vocabulary(
            opts.vocabulary[self.tl], name=self.tl)
        self.trg_vocab = self.vocabs[self.tl]

        # CNN stuff
        opts.model['img_mode'] = kwargs.pop('img_mode', 'raw')
        if opts.model['img_mode'] == 'raw':
            assert 'cnn_type' in opts.model, "cnn_type not provided"
            assert 'cnn_layer' in opts.model, "cnn_layer not provided"
            opts.model['cnn_trainable'] = kwargs.pop('cnn_trainable', False)
            opts.model['cnn_pretrained'] = kwargs.pop('cnn_pretrained', True)
            opts.model['cnn_type'] = kwargs.pop('cnn_type', 'vgg19')
            opts.model['cnn_layer'] = kwargs.pop('cnn_layer', 'conv5_4')
        else:
            # Make this fail to make it mandatory
            assert 'ctx_dim' in opts.model, \
                "You should provide ctx_dim for the given conv features."

            opts.model['ctx_dim'] = kwargs.pop('ctx_dim')

        opts.model['emb_dim'] = kwargs.pop('emb_dim', 100)
        opts.model['dec_dim'] = kwargs.pop('dec_dim', 1000)
        opts.model['dropout'] = kwargs.pop('dropout', 0.5)
        opts.model['prev2out'] = kwargs.pop('prev2out', True)
        opts.model['ctx2out'] = kwargs.pop('ctx2out', True)
        opts.model['selector'] = kwargs.pop('selector', True)
        opts.model['alpha_c'] = kwargs.pop('alpha_c', 1.0)

        # Sanity check after consuming all arguments
        if len(kwargs) > 0:
            self.print('Unused model args: {}'.format(','.join(kwargs.keys())))

        self.opts = opts
        # Need to be set for early-stop evaluation
        self.val_refs = self.opts.data['val_set'][self.tl]

    def train(self, mode=True):
        """Override to customize behaviour."""
        self.training = mode
        for module in self.children():
            module.train(mode)
        if self.opts.model['img_mode'] == 'raw' and \
                not self.opts.model['cnn_trainable']:
            self.cnn.train(False)

    def setup(self, reset_params=True):
        """Sets up NN topology by creating the layers."""
        def process_raw_image(image):
            # Extract convolutional features
            # Section 3.1: a = {a_1 ... a_L}, a_i \in \mathrm{R}^D
            img_ctx = self.cnn(image).view((image.shape[0],
                                            self.opts.model['ctx_dim'], -1))
            # make them (batch_size, w*h, ctx_dim)
            img_ctx.transpose_(1, 2)
            return img_ctx

        # Get CNN encoder for the images
        if self.opts.model['img_mode'] == 'raw':
            self.print('Loading CNN')
            cnn_encoder = ImageEncoder(
                cnn_type=self.opts.model['cnn_type'],
                pretrained=self.opts.model['cnn_pretrained'])
            self.cnn, dims = cnn_encoder.get(self.opts.model['cnn_layer'])
            self.opts.model['ctx_dim'] = dims[1]
            set_learnable(self.cnn, self.opts.model['cnn_trainable'])
            self._process_image = process_raw_image
        else:
            # no-op as features are provided as inputs directly
            self._process_image = lambda x: x

        if self.opts.model['dropout'] > 0:
            # Add 3 dropouts following the original implementation
            self.dropout_img_ctx = nn.Dropout(p=self.opts.model['dropout'])
            self.dropout_lstm = nn.Dropout(p=self.opts.model['dropout'])
            self.dropout_logit = nn.Dropout(p=self.opts.model['dropout'])

        # FF's for LSTM's initial h0, c0
        self.ff_init_c0 = FF(self.opts.model['ctx_dim'],
                             self.opts.model['dec_dim'],
                             bias_zero=True, activ='tanh')
        self.ff_init_h0 = FF(self.opts.model['ctx_dim'],
                             self.opts.model['dec_dim'],
                             bias_zero=True, activ='tanh')

        # Create embedding layer
        self.emb = nn.Embedding(len(self.trg_vocab),
                                self.opts.model['emb_dim'], padding_idx=0)

        # Decoder RNN
        # Create an LSTM from [y_t, z_t] to dec_dim
        lstm_input_size = self.opts.model['ctx_dim'] + \
            self.opts.model['emb_dim']

        self.decoder = nn.LSTMCell(input_size=lstm_input_size,
                                   hidden_size=self.opts.model['dec_dim'])

        # Soft Attention
        self.ff_att = Attention(self.opts.model['ctx_dim'],
                                self.opts.model['dec_dim'])

        # Gating Scalar, i.e. selector
        if self.opts.model['selector']:
            self.ff_selector = FF(self.opts.model['dec_dim'], 1,
                                  bias_zero=True, activ='sigmoid')

        # Output Logic
        self.ff_out_lstm = FF(self.opts.model['dec_dim'],
                              self.opts.model['emb_dim'], bias_zero=True)

        if self.opts.model['ctx2out']:
            self.ff_out_ctx = FF(self.opts.model['ctx_dim'],
                                 self.opts.model['emb_dim'], bias_zero=True)

        self.ff_pre_softmax = FF(self.opts.model['emb_dim'],
                                 len(self.trg_vocab), bias_zero=True)

    def load_data(self, split):
        """Loads the requested dataset split."""
        if split not in self.datasets:
            dataset = Multi30kDataset(split=split,
                                      img_mode=self.opts.model['img_mode'],
                                      data_dict=self.opts.data,
                                      vocabs=self.vocabs,
                                      topology=self.topology,
                                      logger=self.logger)
            self.datasets[split] = dataset
            self.print(dataset)

    def f_init(self, batch):
        # Process image
        img_ctx = self._process_image(batch['image'])

        # Compute mean image context -> (batch_size, _ctx_dim)
        mean_ctx = img_ctx.mean(dim=1)

        if self.opts.model['dropout']:
            mean_ctx = self.dropout_img_ctx(mean_ctx)

        # Compute initial cell state and hidden state for LSTM
        c_t = self.ff_init_c0(mean_ctx)
        h_t = self.ff_init_h0(mean_ctx)

        return (img_ctx, c_t, h_t)

    def f_next(self, img_ctx, y_t, c_t, h_t):
        """Defines the inner block of recurrence."""
        # Apply attention
        alpha_t, z_t = self.ff_att(img_ctx, h_t)

        if self.opts.model['alpha_c']:
            # Append (batch_size x 196) attention probabilities
            self.alphas.append(alpha_t.squeeze(1))

        # Apply selector
        if self.opts.model['selector']:
            z_t *= self.ff_selector(h_t)

        # Form LSTM input by concatenating (embs[:, t], z_t)
        dec_inp = torch.cat([y_t, z_t], dim=1)

        # Get next state
        h_t, c_t = self.decoder(dec_inp, (h_t, c_t))
        if self.opts.model['dropout']:
            h_t = self.dropout_lstm(h_t)

        # This h_t, (optionally along with embs and z_t)
        # will connect to softmax() predictions.
        logit = self.ff_out_lstm(h_t)

        if self.opts.model['prev2out']:
            logit += y_t

        if self.opts.model['ctx2out']:
            logit += self.ff_out_ctx(z_t)

        # Unnormalized vocabulary scores
        logit = F.tanh(logit)
        if self.opts.model['dropout']:
            logit = self.dropout_logit(logit)
        logit = self.ff_pre_softmax(logit)

        # Compute softmax
        log_p = -F.log_softmax(logit, dim=1)

        return log_p, c_t, h_t, alpha_t

    def forward(self, batch):
        """Forward method receives target-length ordered batches."""

        # Encode image and get initial variables
        img_ctx, c_t, h_t = self.f_init(batch)

        # Fetch embeddings -> (seq_len, batch_size, emb_dim)
        caption = batch[self.tl]

        # n_tokens token processed in this batch
        self.n_tokens = caption.numel()

        # Get embeddings
        embs = self.emb(caption)

        # Accumulators
        loss = 0.0
        self.alphas = []

        # -1: So that we skip the timestep where input is <eos>
        for t in range(caption.shape[0] - 1):
            # NOTE: This is where scheduled sampling will happen
            # Either fetch from self.emb or from log_p
            # Current textual input to decoder: y_t = embs[t]
            log_p, c_t, h_t, _ = self.f_next(img_ctx, embs[t], c_t, h_t)

            # t + 1: We're predicting next token
            # Cumulate losses
            loss += torch.gather(
                log_p, dim=1, index=caption[t + 1].unsqueeze(1)).sum()

        if self.opts.model['alpha_c']:
            self.aux_loss = self.opts.model['alpha_c'] * \
                ((1 - sum(self.alphas))**2).sum(0).mean()

        # Return normalized loss
        return loss / self.n_tokens

    def compute_loss(self, data_loader):
        r"""Computes test set loss over the given data_loader instance."""
        n_tokens_seen = 0
        total_loss = 0.0

        for batch in data_loader:
            loss = self.forward(to_var(batch, volatile=True))
            total_loss += (loss * self.n_tokens)
            n_tokens_seen += self.n_tokens

        return total_loss.data.cpu()[0] / n_tokens_seen

    def beam_search(self, data_loader, k=12, max_len=100,
                    avoid_double=False, avoid_unk=False):
        """Performs beam search over split and returns the hyps."""
        results = []
        inf = 1e3
        bos = self.trg_vocab['<bos>']
        eos = self.trg_vocab['<eos>']
        n_tokens = len(self.trg_vocab)

        for batch in pbar(data_loader, unit='batch'):
            n = batch.size

            # Mask to apply to pdxs.view(-1) to fix indices
            nk_mask = torch.arange(n * k).long().cuda()
            pdxs_mask = (nk_mask / k) * k

            # We can fill this to represent the beams in tensor format
            beam = torch.zeros((max_len, n, k)).long().cuda()

            # Get initial image context and LSTM states
            # ic: N x C, c_t: N x H, h_t: N x H
            ict, c_t, h_t = self.f_init(to_var(batch, volatile=True))

            # Initial y_t for <bos> embs: N x emb_dim
            y_t = self.emb(to_var(
                torch.ones(n).long() * bos, volatile=True))

            log_p, c_t, h_t, _ = self.f_next(ict, y_t, c_t, h_t)
            nll, beam[0] = log_p.data.topk(k, 1, sorted=False, largest=False)

            # Tile indices to use in the loop to expand first dim
            tile = nk_mask / k

            for t in range(1, max_len):
                cur_tokens = beam[t - 1].view(-1)
                fini_idxs = (cur_tokens == eos).nonzero()
                n_fini = fini_idxs.numel()
                if n_fini == n * k:
                    break

                # Fetch embs for the next iteration (N*B, E)
                y_t = self.emb(to_var(cur_tokens, volatile=True))

                # Get log_probs and new LSTM states (log_p, N*B, V)
                ict = ict[tile]
                log_p, c_t, h_t, _ = self.f_next(ict, y_t, c_t[tile], h_t[tile])
                log_p = log_p.data

                # Suppress probabilities of previous tokens
                if avoid_double:
                    log_p.view(-1).index_fill_(
                        0, cur_tokens + (nk_mask * n_tokens), inf)

                # Avoid <unk> tokens
                if avoid_unk:
                    log_p[:, self.trg_vocab['<unk>']] = inf

                # Favor finished hyps to generate <eos> again
                # Their nll scores will not increase further and they will
                # always be kept in the beam.
                if n_fini > 0:
                    fidxs = fini_idxs[:, 0]
                    log_p.index_fill_(0, fidxs, inf)
                    log_p.view(-1).index_fill_(0, fidxs * n_tokens + eos, 0)

                # Expand to 3D, cross-sum scores and reduce back to 2D
                nll = (nll.unsqueeze(2) + log_p.view(n, k, -1)).view(n, -1)

                # Reduce (n, k*v) to k-best
                nll, idxs = nll.topk(k, 1, sorted=False, largest=False)

                # previous indices into the beam and current token indices
                pdxs = idxs / n_tokens

                # Insert current tokens
                beam[t] = idxs % n_tokens

                # Permute all hypothesis history according to new order
                beam[:t] = beam[:t].gather(2, pdxs.repeat(t, 1, 1))

                # Compute correct previous indices
                # Mask is needed since we're in flattened regime
                tile = pdxs.view(-1) + pdxs_mask

            # Put an explicit <eos> to make idxs_to_sent happy
            beam[max_len - 1] = eos

            # Find lengths by summing tokens not in (pad,bos,eos)
            lens = (beam.transpose(0, 2) > 2).sum(-1).t().float().clamp(min=1)
            # Normalize scores by length
            nll /= lens.float()
            top_hyps = nll.topk(1, sorted=False, largest=False)[1].squeeze(1)
            # Get best hyp for each sample in the batch
            hyps = beam[:, range(n), top_hyps].cpu().numpy().T
            results.extend(self.trg_vocab.list_of_idxs_to_sents(hyps))

        return results
