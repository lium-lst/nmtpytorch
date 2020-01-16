# -*- coding: utf-8 -*-
from ..logger import Logger

import torch
from torch import nn
from torch import distributions
from torch.nn import functional as F

from ..layers import FF, ArgSelect, PEmbedding
from ..layers.decoders import ConditionalDecoder
from ..utils.nn import get_rnn_hidden_state
from ..datasets import MultimodalDataset
from ..utils.misc import pbar
from ..utils.topology import Topology
from ..utils.device import DEVICE
from ..utils.data import sort_predictions

from . import NMT

log = Logger()


class VectorDecoder(ConditionalDecoder):
    """Single-layer RNN decoder using fixed-size vector representation."""
    def __init__(self, **kwargs):
        # Disable attention
        kwargs['att_type'] = None
        super().__init__(**kwargs)

        # Keep losses per sample
        del self.nll_loss
        self.nll_loss = nn.NLLLoss(reduction='none', ignore_index=0)

    def f_next(self, ctx_dict, y, h):
        """Applies one timestep of recurrence."""
        # Get hidden states from the decoder
        h1_c1 = self.dec0(y, self._rnn_unpack_states(h))
        h1 = get_rnn_hidden_state(h1_c1)

        # Project hidden state to embedding size
        o = self.hid2out(h1)

        # Apply dropout if any
        logit = self.do_out(o) if self.dropout_out > 0 else o

        # Transform logit to T*B*V (V: vocab_size)
        # Compute log_softmax over token dim
        log_p = F.log_softmax(self.out2prob(logit), dim=-1)

        # Return log probs and new hidden states
        return log_p, self._rnn_pack_states(h1_c1)

    def forward(self, ctx_dict, y):
        """Computes the softmax outputs given source annotations `ctx_dict[self.ctx_name]`
        and ground-truth target token indices `y`. Only called during training.

        Arguments:
            ctx_dict(dict): A dictionary of tensors that should at least contain
                the key `ctx_name` as the main source representation of shape
                S*B*ctx_dim`.
            y(Tensor): A tensor of `T*B` containing ground-truth target
                token indices for the given batch.
        """

        losses = []

        # Get initial hidden state
        h = self.f_init(ctx_dict)

        # Convert token indices to embeddings -> T*B*E
        # Skip <bos> now
        bos = self.get_emb(y[0], 0)
        log_p, h = self.f_next(ctx_dict, bos, h)
        losses.append(self.nll_loss(log_p, y[1]))

        y_emb = self.get_emb(y[1:])

        for t in range(y_emb.shape[0] - 1):
            log_p, h = self.f_next(ctx_dict, y_emb[t], h)
            losses.append(self.nll_loss(log_p, y[t + 2]))

        # T x B
        losses = torch.stack(losses)

        return {'loss': losses.sum(), 'losses': losses}


class Rationalev3(NMT):
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            # Generator
            'gen_dim': 256,             # Generator hidden size
            'gen_type': 'gru',          # Generator type (gru|lstm|emb)
            'gen_bidir': True,          # Bi-directional encoder in generator
            'gen_n_layers': 1,          # Number of stacked encoders in the generator
            'gen_nolearn': False,       # Random masked baseline with reinforce loss (debug)
            # Encoder
            'enc_dim': 256,             # Re-encoder hidden size
            'enc_type': 'gru',          # Encoder type (gru|lstm|avg|sum|max)
            'enc_bidir': True,          # Bi-directional encoder in re-encoder
            'enc_n_layers': 1,          # If enc_type is recurrent: # of layers
            # Decoder
            'dec_dim': 256,             # Decoder hidden size
            'dec_type': 'gru',          # Decoder type (gru|lstm)
            'dec_variant': 'vector',    # (cond|simplegru|vector)
            'dec_init': 'mean_ctx',     # How to initialize decoder (zero/mean_ctx/feats)
            # Generic S2S arguments
            'direction': None,          # Network directionality, i.e. en->de
            'max_len': 80,              # Reject sentences where 'bucket_by' length > 80
            'bucket_by': None,          # A key like 'en' to define w.r.t which dataset
                                        # the batches will be sorted
            'bucket_order': None,       # Curriculum: ascending/descending/None
            'sampler_type': 'bucket',   # bucket or approximate
            # Other arguments
            'emb_dim': 128,             # Source and target embedding sizes
            'proj_emb_dim': 128,        # Additional embedding projection output
            'proj_emb_activ': 'linear',  # Additional embedding projection non-lin
            'pretrained_embs': '',      # optional path to embeddings tensor
            'pretrained_embs_l2': False,  # L2 normalize pretrained embs
            'short_list': 100000000,    # Use all pretrained vocab
            'dropout': 0,               # Simple dropout layer
            'add_src_eos': True,        # Append <eos> or not to inputs
            'rnn_dropout': 0,           # RNN dropout if *_n_layers > 1
            'mode': 'autoencoder',      # autoencoder|reinforce|gumbel
            'lambda_sparsity': 0.0003,  # sparsity factor
            'lambda_coherence': 0.0006,  # coherency factor
            'reinforce_lr': 10,         # Learning rate for reinforce gradient
            'reinforce_samples': 1,     # How many samples to get from generator
            'tied_emb': False,          # Not used, always tied in this model
        }

    def __init__(self, opts):
        super().__init__(opts)

        # Clip to short_list if given
        self.n_src_vocab = min(self.n_src_vocab, self.opts.model['short_list'])
        self.n_trg_vocab = min(self.n_trg_vocab, self.opts.model['short_list'])

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        self.dataset = MultimodalDataset(
            data=self.opts.data['{}_set'.format(split)],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model['max_len'],
            bucket_order=self.opts.model['bucket_order'],
            sampler_type=self.opts.model['sampler_type'],
            eos=self.opts.model['add_src_eos'])
        log.log(self.dataset)
        return self.dataset

    def reset_parameters(self):
        for name, param in self.named_parameters():
            # Skip 1-d biases and scalars
            if param.requires_grad and param.dim() > 1:
                nn.init.kaiming_normal_(param.data)

        with torch.no_grad():
            if self.opts.model['pretrained_embs']:
                embs = torch.load(self.opts.model['pretrained_embs'])
                embs = embs[:self.opts.model['short_list']].float()
                # Reset padding embedding to 0
                embs[0] = 0
                if self.opts.model['pretrained_embs_l2']:
                    embs.div_(embs.norm(p=2, dim=-1, keepdim=True))
                self.embs.weight.data.copy_(embs)
                self.dec.emb.weight.data.copy_(embs)

    def setup(self, is_train=True):
        """Sets up NN topology by creating the layers."""
        # Create embeddings followed by projection layer
        self.embs = PEmbedding(
            self.n_src_vocab, self.opts.model['emb_dim'],
            out_dim=self.opts.model['proj_emb_dim'],
            activ=self.opts.model['proj_emb_activ'])

        # Generic dropout layer
        self.do = nn.Dropout(self.opts.model['dropout'])

        ##################
        # Generator block
        ##################
        layers = []
        gen_type = self.opts.model['gen_type'].upper()
        if gen_type in ('GRU', 'LSTM'):
            RNN = getattr(nn, gen_type)

            # RNN Encoder
            layers.append(RNN(
                self.opts.model['proj_emb_dim'], self.opts.model['gen_dim'],
                self.opts.model['gen_n_layers'], batch_first=False,
                dropout=self.opts.model['rnn_dropout'],
                bidirectional=self.opts.model['gen_bidir']))
            # Return the sequence of hidden outputs
            layers.append(ArgSelect(0))

            # Consider bi-directionality
            self.gen_out = self.opts.model['gen_dim'] * (int(self.opts.model['gen_bidir']) + 1)

            # Create the generator wrapper
            self.gen = nn.Sequential(*layers)
        elif gen_type == 'EMB':
            # Directly go from embeddings without processing with RNN
            self.gen_out = self.opts.model['proj_emb_dim']
            self.gen = lambda x: x

        # If we don't have any re-encoder, the source encoding dim is gen_out
        self.ctx_size = self.gen_out

        ############################
        # Independent selector layer
        ############################
        if self.opts.model['mode'] == 'reinforce':
            self.reinforce_samples = torch.ones(
                (self.opts.model['reinforce_samples'], 1, 1))
            self.z_layer = FF(self.gen_out, 1, activ='sigmoid')

        ###################
        # Create Re-encoder
        ###################
        layers = []
        enc_type = self.opts.model['enc_type'].upper()
        if enc_type in ('GRU', 'LSTM'):
            RNN = getattr(nn, enc_type)

            # RNN Re-encoder
            layers.append(RNN(
                self.opts.model['proj_emb_dim'], self.opts.model['enc_dim'],
                self.opts.model['gen_n_layers'], batch_first=False,
                dropout=self.opts.model['rnn_dropout'],
                bidirectional=self.opts.model['gen_bidir']))
            # Return the sequence of hidden outputs
            layers.append(ArgSelect(0))

            # Create the re-encoder wrapper
            self.enc = nn.Sequential(*layers)

            self.ctx_size = self.opts.model['enc_dim'] * int(self.opts.model['enc_bidir'] + 1)
        else:
            # Poolings will be done in the decoder
            self.enc = lambda x: x

        ################
        # Create Decoder
        ################
        self.dec = VectorDecoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict={str(self.sl): self.ctx_size},
            ctx_name=str(self.sl),
            tied_emb=True,
            dec_init=self.opts.model['dec_init'],
            dropout_out=self.opts.model['dropout'])

    def encode(self, batch, **kwargs):
        # Fetch embeddings -> T x B x D
        embs = self.embs(batch[str(self.sl)])
        do_embs = self.do(embs)

        # Dropout and pass embeddings through generator -> T x B x G
        hs = self.gen(do_embs)

        if self.opts.model['mode'] == 'reinforce':
            # Apply a sigmoid layer and transpose -> B x T
            p_z = self.z_layer(hs).squeeze(-1).t()

            # Create a distribution
            self.dist = distributions.Binomial(
                total_count=self.reinforce_samples, probs=p_z)

            # Draw N samples from it -> N x B x T
            self.z = self.dist.sample()

            # Mask out non-rationale bits and re-encode -> 1, B, G
            sent_rep = self.enc(self.z.permute((2, 1, 0)) * do_embs)

            return {str(self.sl): (sent_rep, None), 'z': (self.z, None)}

        elif self.opts.model['mode'] == 'autoencoder':
            # Use the output of the generator directly
            return {str(self.sl): (hs, None)}

    def forward(self, batch, **kwargs):
        result = self.dec(self.encode(batch), batch[self.tl])
        result['n_items'] = torch.nonzero(batch[self.tl][1:]).shape[0]

        if self.opts.model['mode'] == 'reinforce' and not self.opts.model['gen_nolearn']:
            # z-> TxB
            z = self.z.squeeze(0).t()
            # per instance rationale word counts
            zsum = z.sum(0)
            # continuity reward (probably not relevant for us)
            zdif = (z[1:] - z[:-1]).abs().sum(0)

            # Per instance cross-entropy losses (T x B) -> (B, )
            rewards = result.pop('losses').detach().mean(0)

            # squeeze will work for 1-sample case
            per_instance_log_prob = -self.dist.log_prob(self.z).mean(-1).squeeze()

            # Enrich the rewards with regularization terms
            scaled_zsum = self.opts.model['lambda_sparsity'] * zsum
            scaled_zdif = self.opts.model['lambda_coherence'] * zdif
            rewards.add_(scaled_zsum + scaled_zdif)

            self.aux_loss['gen_loss'] = self.opts.model['reinforce_lr'] * \
                (rewards * per_instance_log_prob).mean()

            # Log some stuff to tensorboard
            if self.training and kwargs['uctr'] % self.opts.train['disp_freq'] == 0:
                self.tboard.log_scalar('z_sum', z.mean().item(), kwargs['uctr'])

        return result

    @staticmethod
    def beam_search_dump(models, data_loader, task_id=None, beam_size=12,
                         max_len=200, lp_alpha=0., suppress_unk=False, n_best=False):
        """An efficient implementation for beam-search algorithm.

        Arguments:
            models (list of Model): Model instance(s) derived from `nn.Module`
                defining a set of methods. See `models/nmt.py`.
            data_loader (DataLoader): A ``DataLoader`` instance.
            task_id (str, optional): For multi-output models, this selects
                the decoder. (Default: None)
            beam_size (int, optional): The size of the beam. (Default: 12)
            max_len (int, optional): Maximum target length to stop beam-search
                if <eos> is still not generated. (Default: 200)
            lp_alpha (float, optional): If > 0, applies Google's length-penalty
                normalization instead of simple length normalization.
                lp: ((5 + |Y|)^lp_alpha / (5 + 1)^lp_alpha)
            suppress_unk (bool, optional): If `True`, suppresses the log-prob
                of <unk> token.
            n_best (bool, optional): If `True`, returns n-best list of the beam
                with the associated scores.

        Returns:
            list:
                A list of hypotheses in surface form.
        """

        def tile_ctx_dict(ctx_dict, idxs):
            """Returns dict of 3D tensors repeatedly indexed along the sample axis."""
            # 1st: tensor, 2nd optional mask
            return {
                k: (t[:, idxs], None if mask is None else mask[:, idxs])
                for k, (t, mask) in ctx_dict.items()
            }

        def check_context_ndims(ctx_dict):
            for name, (ctx, mask) in ctx_dict.items():
                assert ctx.dim() == 3, \
                    f"{name}'s 1st dim should always be a time dimension."

        # This is the batch-size requested by the user but with sorted
        # batches, efficient batch-size will be <= max_batch_size
        max_batch_size = data_loader.batch_sampler.batch_size
        k = beam_size
        inf = -1000
        results = []
        final_hyps = []
        src_sents = []
        enc_args = {}

        if task_id is None:
            # For classical models that have single encoder, decoder and
            # target vocabulary
            decs = [m.dec for m in models]
            f_inits = [dec.f_init for dec in decs]
            f_nexts = [dec.f_next for dec in decs]
            vocab = models[0].trg_vocab
        else:
            # A specific input-output topology has been requested
            task = Topology(task_id)
            enc_args['enc_ids'] = task.srcs
            # For new multi-target models: select the first target decoder
            decs = [m.get_decoder(task.first_trg) for m in models]
            # Get the necessary init() and next() methods
            f_inits = [dec.f_init for dec in decs]
            f_nexts = [dec.f_next for dec in decs]
            # Get the corresponding vocabulary for the first target
            vocab = models[0].vocabs[task.first_trg]

        # Common parts
        encoders = [m.encode for m in models]
        unk = vocab['<unk>']
        eos = vocab['<eos>']
        n_vocab = len(vocab)

        # Tensorized beam that will shrink and grow up to max_batch_size
        beam_storage = torch.zeros(
            max_len, max_batch_size, k, dtype=torch.long, device=DEVICE)
        mask = torch.arange(max_batch_size * k, device=DEVICE)
        nll_storage = torch.zeros(max_batch_size, device=DEVICE)

        for batch in pbar(data_loader, unit='batch'):
            batch.device(DEVICE)

            # Always use the initial storage
            beam = beam_storage.narrow(1, 0, batch.size).zero_()

            # Mask to apply to pdxs.view(-1) to fix indices
            nk_mask = mask.narrow(0, 0, batch.size * k)

            # nll: batch_size x 1 (will get expanded further)
            nll = nll_storage.narrow(0, 0, batch.size).unsqueeze(1)

            # Tile indices to use in the loop to expand first dim
            tile = range(batch.size)

            # Encode source modalities
            ctx_dicts = [encode(batch, **enc_args) for encode in encoders]

            ###############################################
            # NOTE: Hack to dump masks for rationale model
            ###############################################
            src_idxs = batch[models[0].sl].cpu().t().tolist()
            s_masks = ctx_dicts[0]['z'][0].squeeze(0).long().cpu().tolist()
            get_word = lambda i, m: vocab[i] if m else f'[{vocab[i]}]'
            for s_idxs, s_mask in zip(src_idxs, s_masks):
                src_sents.append(
                    ' '.join([get_word(idx, m) for idx, m in zip(s_idxs, s_mask)]))
            ###############################################

            # Sanity check one of the context dictionaries for dimensions
            check_context_ndims(ctx_dicts[0])

            # Get initial decoder state (N*H)
            h_ts = [f_init(ctx_dict) for f_init, ctx_dict in zip(f_inits, ctx_dicts)]

            # we always have <bos> tokens except that the returned embeddings
            # may differ from one model to another.
            idxs = models[0].get_bos(batch.size).to(DEVICE)

            for tstep in range(max_len):
                # Select correct positions from source context
                ctx_dicts = [tile_ctx_dict(cd, tile) for cd in ctx_dicts]

                # Get log probabilities and next state
                # log_p: batch_size x vocab_size (t = 0)
                #        batch_size*beam_size x vocab_size (t > 0)
                # NOTE: get_emb does not exist in some models, fix this.
                log_ps, h_ts = zip(
                    *[f_next(cd, dec.get_emb(idxs, tstep), h_t[tile]) for
                      f_next, dec, cd, h_t in zip(f_nexts, decs, ctx_dicts, h_ts)])

                # Do the actual averaging of log-probabilities
                log_p = sum(log_ps).data

                if suppress_unk:
                    log_p[:, unk] = inf

                # Detect <eos>'d hyps
                idxs = (idxs == 2).nonzero()
                if idxs.numel():
                    if idxs.numel() == batch.size * k:
                        break
                    idxs.squeeze_(-1)
                    # Unfavor all candidates
                    log_p.index_fill_(0, idxs, inf)
                    # Favor <eos> so that it gets selected
                    log_p.view(-1).index_fill_(0, idxs * n_vocab + 2, 0)

                # Expand to 3D, cross-sum scores and reduce back to 2D
                # log_p: batch_size x vocab_size ( t = 0 )
                #   nll: batch_size x beam_size (x 1)
                # nll becomes: batch_size x beam_size*vocab_size here
                # Reduce (N, K*V) to k-best
                nll, beam[tstep] = nll.unsqueeze_(2).add(log_p.view(
                    batch.size, -1, n_vocab)).view(batch.size, -1).topk(
                        k, sorted=False, largest=True)

                # previous indices into the beam and current token indices
                pdxs = beam[tstep] / n_vocab
                beam[tstep].remainder_(n_vocab)
                idxs = beam[tstep].view(-1)

                # Compute correct previous indices
                # Mask is needed since we're in flattened regime
                tile = pdxs.view(-1) + (nk_mask / k) * (k if tstep else 1)

                if tstep > 0:
                    # Permute all hypothesis history according to new order
                    beam[:tstep] = beam[:tstep].gather(2, pdxs.repeat(tstep, 1, 1))

            # Put an explicit <eos> to make idxs_to_sent happy
            beam[max_len - 1] = eos

            # Find lengths by summing tokens not in (pad,bos,eos)
            len_penalty = beam.gt(2).float().sum(0).clamp(min=1)

            if lp_alpha > 0.:
                len_penalty = ((5 + len_penalty)**lp_alpha) / 6**lp_alpha

            # Apply length normalization
            nll.div_(len_penalty)

            if n_best:
                # each elem is sample, then candidate
                tbeam = beam.permute(1, 2, 0).to('cpu').tolist()
                scores = nll.to('cpu').tolist()
                results.extend(
                    [(vocab.list_of_idxs_to_sents(b), s) for b, s in zip(tbeam, scores)])
            else:
                # Get best-1 hypotheses
                top_hyps = nll.topk(1, sorted=False, largest=True)[1].squeeze(1)
                hyps = beam[:, range(batch.size), top_hyps].t().to('cpu')
                hyps_list = vocab.list_of_idxs_to_sents(hyps.tolist())
                final_hyps.extend(hyps_list)

        # Recover order of the samples if necessary
        for srcsent, finalhyp in zip(src_sents, final_hyps):
            results.append(f'{srcsent} ||| {finalhyp}')
        return sort_predictions(data_loader, results)
