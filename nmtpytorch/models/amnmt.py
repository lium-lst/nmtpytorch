# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vocabulary import Vocabulary
from ..layers import ImageEncoder, TextEncoder, ConditionalMMDecoder
from ..utils.nn import get_network_topology, set_learnable
from ..utils.data import to_var
from ..utils.misc import pbar

from ..datasets import Multi30kDataset


class AttentiveMNMT(nn.Module):
    r"""A sequence-to-sequence NMT model with visual attention."""
    def __init__(self, opts, logger=None):
        """This should only be used for argument processing."""
        super(AttentiveMNMT, self).__init__()

        self.vocabs = {}
        self.datasets = {}
        self.logger = logger
        self.print = print if self.logger is None else self.logger.info

        # Get a copy of model options coming from config
        # Consume them with .pop(), integrating model defaults
        # What is left inside kwargs -> Unused arguments
        kwargs = opts.model.copy()

        # Get direction and parse it
        opts.model['direction'] = kwargs.pop('direction')
        self.topo = get_network_topology(opts.model['direction'])

        # Load vocabularies here
        for lang in self.topo['src_langs'] + self.topo['trg_langs']:
            self.vocabs[lang] = Vocabulary(opts.vocabulary[lang])

        self.sl = self.topo['src_langs'][0]
        self.tl = self.topo['trg_langs'][0]

        self.src_vocab = self.vocabs[self.sl]
        self.trg_vocab = self.vocabs[self.tl]

        self.n_src_vocab = len(self.src_vocab)
        self.n_trg_vocab = len(self.trg_vocab)

        # CNN stuff
        opts.model['img_mode'] = kwargs.pop('img_mode', 'raw')
        if opts.model['img_mode'] == 'raw':
            assert 'cnn_type' in opts.model, "cnn_type not provided"
            assert 'cnn_layer' in opts.model, "cnn_layer not provided"
            opts.model['cnn_trainable'] = kwargs.pop('cnn_trainable', False)
            opts.model['cnn_pretrained'] = kwargs.pop('cnn_pretrained', True)
            opts.model['cnn_type'] = kwargs.pop('cnn_type', 'resnet50')
            opts.model['cnn_layer'] = kwargs.pop('cnn_layer', 'res4f_relu')
        else:
            # Make this fail to make it mandatory
            assert 'img_ctx_dim' in opts.model, \
                "You should provide img_ctx_dim for the given conv features."

            opts.model['img_ctx_dim'] = kwargs.pop('img_ctx_dim')

        # How to fusion multimodal contexts (sum/mul/concat)
        opts.model['fusion_type'] = kwargs.pop('fusion_type', 'sum')

        opts.model['emb_dim'] = kwargs.pop('emb_dim', 256)
        opts.model['enc_dim'] = kwargs.pop('enc_dim', 512)
        opts.model['dec_dim'] = kwargs.pop('dec_dim', 1024)
        opts.model['enc_type'] = kwargs.pop('enc_type', 'gru')
        opts.model['dec_type'] = kwargs.pop('dec_type', 'gru')
        opts.model['n_encoders'] = kwargs.pop('n_encoders', 1)

        # How to initialize decoder (zero/mean_ctx)
        opts.model['dec_init'] = kwargs.pop('dec_init', 'mean_ctx')

        # Attention related parameters
        opts.model['att_type'] = kwargs.pop('att_type', 'mlp')

        # The bottleneck dimension during attention computation
        opts.model['att_bottleneck'] = kwargs.pop('att_bottleneck', 'ctx')
        opts.model['att_activ'] = kwargs.pop('att_activ', 'tanh')

        # Various dropouts
        opts.model['dropout_emb'] = kwargs.pop('dropout_emb', 0.)
        opts.model['dropout_ctx'] = kwargs.pop('dropout_ctx', 0.)
        opts.model['dropout_out'] = kwargs.pop('dropout_out', 0.)
        opts.model['dropout_enc'] = kwargs.pop('dropout_enc', 0.)

        # Tie embeddings: False/2way/3way
        opts.model['tied_emb'] = kwargs.pop('tied_emb', False)

        # Sanity check after consuming all arguments
        if len(kwargs) > 0:
            self.print('Unused model args: {}'.format(','.join(kwargs.keys())))

        # Check vocabulary sizes for 3way tying
        if opts.model['tied_emb'] == '3way':
            assert self.n_src_vocab == self.n.trg_vocab, \
                "The vocabulary sizes do not match for 3way tied embeddings."

        # Textual context size is always equal to enc_dim * 2 since
        # it is the concatenation of forward and backward hidden states
        self.txt_ctx_dim = opts.model['enc_dim'] * 2

        # Set some shortcuts
        self.opts = opts

        # Need to be set for early-stop evaluation
        self.val_refs = self.opts.data['val_set'][self.tl]

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad and 'bias' not in name:
                nn.init.kaiming_normal(param.data)

    def setup_cnn(self):
        def process_raw_image(image):
            # Extract convolutional features
            # Section 3.1: a = {a_1 ... a_L}, a_i \in \mathrm{R}^D
            img_ctx = self.cnn(image).view(
                (image.shape[0], self.opts.model['img_ctx_dim'], -1))
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
            self.opts.model['img_ctx_dim'] = dims[1]
            self.img_ctx_dim = self.opts.model['img_ctx_dim']
            self._process_image = process_raw_image

            # Set learnable flag
            set_learnable(self.cnn, self.opts.model['cnn_trainable'])
        else:
            # no-op as features are provided as inputs directly
            self._process_image = lambda x: x

    def setup(self, reset_params=True):
        """Sets up NN topology by creating the layers."""

        # Setup CNN
        self.setup_cnn()

        # Create encoder
        self.enc = TextEncoder(input_size=self.opts.model['emb_dim'],
                               hidden_size=self.opts.model['enc_dim'],
                               n_vocab=self.n_src_vocab,
                               cell_type=self.opts.model['enc_type'],
                               dropout_emb=self.opts.model['dropout_emb'],
                               dropout_ctx=self.opts.model['dropout_ctx'],
                               dropout_rnn=self.opts.model['dropout_enc'],
                               num_layers=self.opts.model['n_encoders'])

        # Create Decoder
        self.dec = ConditionalMMDecoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            ctx_size_dict={'img': self.img_ctx_dim, 'txt': self.txt_ctx_dim},
            fusion_type=self.opts.model['fusion_type'],
            tied_emb=self.opts.model['tied_emb'],
            dec_init=self.opts.model['dec_init'],
            att_type=self.opts.model['att_type'],
            att_activ=self.opts.model['att_activ'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout_out'])

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            self.enc.emb.weight = self.dec.emb.weight

        if reset_params:
            self.reset_parameters()

    def get_iterator(self, split, batch_size, only_source=False):
        """Returns a DataLoader for the requested dataset split."""
        return self.datasets[split].get_iterator(batch_size, only_source)

    def load_data(self, split):
        """Loads the requested dataset split."""
        if split not in self.datasets:
            dataset = Multi30kDataset(split=split,
                                      img_mode=self.opts.model['img_mode'],
                                      data_dict=self.opts.data,
                                      vocabs=self.vocabs,
                                      topology=self.topo,
                                      logger=self.logger)
            self.datasets[split] = dataset
            self.print(dataset)

    def aux_loss(self):
        """Returns the sum of auxiliary losses."""
        return 0.

    def compute_loss(self, data_loader):
        """Computes test set loss over the given DataLoader instance."""
        total_loss = 0.0
        n_tokens_seen = 0

        for batch in data_loader:
            loss = self.forward(to_var(batch, volatile=True))
            total_loss += (loss * self.n_tokens)
            n_tokens_seen += self.n_tokens

        return total_loss.data.cpu()[0] / n_tokens_seen

    def encode(self, data):
        """Encodes all inputs and returns a dictionary.

        Arguments:
            data (dict): A batch of samples with keys designating the source
                and target modalities.

        Returns:
            dict:
                A dictionary where keys are source modalities compatible
                with the data loader and the values are tuples where the
                elements are encodings and masks. The mask can be ``None``
                if the relevant modality does not require a mask.
        """
        return {
            'image': (self._process_image(data['image']).transpose(0, 1), None),
            'txt': self.enc(data[self.sl]),
        }

    def forward(self, data):
        """Computes the forward-pass of the network and returns batch loss.

        Arguments:
            data (dict): A batch of samples with keys designating the source
                and target modalities.

        Returns:
            Variable:
                A scalar loss normalized w.r.t batch size and token counts.

        Note:
            This method should assign the number of target tokens for which
            the loss is computed inside ``self.n_tokens`` in order for the
            ``MainLoop`` to correctly compute epoch loss.
        """

        # Get all source encodings
        ctx_dict = self.encode(data)

        # Save number of target tokens for loss normalization
        self.n_tokens = data[self.tl][1:].numel()

        # Get final loss as a sum
        return self.dec(ctx_dict, data[self.tl]) / self.n_tokens

    @staticmethod
    def tile_ctx_dict(ctx_dict, tile):
        # FIXME: Can be a utility method
        return {
            k: (v[0][:, tile], None if v[1] is None else v[1][:, tile])
            for k, v in ctx_dict.items()
        }

    def beam_search(self, data_loader, k=12, max_len=100,
                    avoid_rep=False, avoid_unk=False):
        """Performs beam search over split and returns the hyps."""
        results = []
        inf = 1e3
        bos = self.trg_vocab['<bos>']
        eos = self.trg_vocab['<eos>']

        for batch in pbar(data_loader, unit='batch'):
            n = batch.size

            # Mask to apply to pdxs.view(-1) to fix indices
            nk_mask = torch.arange(n * k).long().cuda()
            pdxs_mask = (nk_mask / k) * k

            # Tile indices to use in the loop to expand first dim
            tile = nk_mask / k

            # We can fill this to represent the beams in tensor format
            beam = torch.zeros((max_len, n, k)).long().cuda()

            # Encode source sentences into ctxs (S*N*C)
            # Encode image
            ctx_dict = self.encode(to_var(batch, volatile=True))

            # Get initial decoder state (N*H)
            h_t = self.dec.f_init(*ctx_dict['txt'])

            # Initial y_t for <bos> embs: N x emb_dim
            y_t = self.dec.emb(to_var(
                torch.ones(n).long() * bos, volatile=True))

            log_p, h_t = self.dec.f_next(ctx_dict, y_t, h_t)
            nll, beam[0] = log_p.data.topk(k, sorted=False, largest=False)

            for t in range(1, max_len):
                cur_tokens = beam[t - 1].view(-1)
                fini_idxs = (cur_tokens == eos).nonzero()
                n_fini = fini_idxs.numel()
                if n_fini == n * k:
                    break

                # Fetch embs for the next iteration (N*K, E)
                y_t = self.dec.emb(to_var(cur_tokens, volatile=True))

                # Get log_probs and new LSTM states (log_p, N*K, V)
                ctx_dict = self.tile_ctx_dict(ctx_dict, tile)
                log_p, h_t = self.dec.f_next(ctx_dict, y_t, h_t[tile])
                log_p = log_p.data

                # Suppress probabilities of previous tokens
                if avoid_rep:
                    log_p.view(-1).index_fill_(
                        0, cur_tokens + (nk_mask * self.n_trg_vocab), inf)

                # Avoid <unk> tokens
                if avoid_unk:
                    log_p[:, self.trg_vocab['<unk>']] = inf

                # Favor finished hyps to generate <eos> again
                # Their nll scores will not increase further and they will
                # always be kept in the beam.
                if n_fini > 0:
                    fidxs = fini_idxs[:, 0]
                    log_p.index_fill_(0, fidxs, inf)
                    log_p.view(-1).index_fill_(
                        0, fidxs * self.n_trg_vocab + eos, 0)

                # Expand to 3D, cross-sum scores and reduce back to 2D
                nll = (nll.unsqueeze(2) + log_p.view(n, k, -1)).view(n, -1)

                # Reduce (N, K*V) to k-best
                nll, idxs = nll.topk(k, sorted=False, largest=False)

                # previous indices into the beam and current token indices
                pdxs = idxs / self.n_trg_vocab

                # Insert current tokens
                beam[t] = idxs % self.n_trg_vocab

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
