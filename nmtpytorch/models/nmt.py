# -*- coding: utf-8 -*-
import logging

import torch
from torch import nn

from ..layers import TextEncoder
from ..layers.decoders import get_decoder
from ..utils.misc import get_n_params
from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from ..utils.ml_metrics import Loss
from ..utils.device import DeviceManager
from ..utils.misc import pbar
from ..utils.data import sort_predictions
from ..datasets import MultimodalDataset
from ..metrics import Metric

import pdb

logger = logging.getLogger('nmtpytorch')


class NMT(nn.Module):
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            'emb_dim': 128,             # Source and target embedding sizes
            'emb_maxnorm': None,        # Normalize embeddings l2 norm to 1
            'emb_gradscale': False,     # Scale embedding gradients w.r.t. batch frequency
            'enc_dim': 256,             # Encoder hidden size
            'enc_type': 'gru',          # Encoder type (gru|lstm)
            'enc_lnorm': False,         # Add layer-normalization to encoder output
            'enc_bidirectional': True,  # Whether the RNN encoder should be bidirectional
            'n_encoders': 1,            # Number of stacked encoders
            'dec_dim': 256,             # Decoder hidden size
            'dec_type': 'gru',          # Decoder type (gru|lstm)
            'dec_variant': 'cond',      # (cond|simplegru|vector)
            'dec_init': 'mean_ctx',     # How to initialize decoder (zero/mean_ctx/feats)
            'dec_init_size': None,      # feature vector dimensionality for
            'dec_init_activ': 'tanh',   # Decoder initialization activation func
                                        # dec_init == 'feats'
            'att_type': 'mlp',          # Attention type (mlp|dot)
            'att_temp': 1.,             # Attention temperature
            'att_activ': 'tanh',        # Attention non-linearity (all torch nonlins)
            'att_mlp_bias': False,      # Enables bias in attention mechanism
            'att_bottleneck': 'ctx',    # Bottleneck dimensionality (ctx|hid)
            'att_transform_ctx': True,  # Transform annotations before attention
            'att_ctx2hid': True,        # Add one last FC layer on top of the ctx
            'dropout_emb': 0,           # Simple dropout to source embeddings
            'dropout_ctx': 0,           # Simple dropout to source encodings
            'dropout_out': 0,           # Simple dropout to decoder output
            'dropout_enc': 0,           # Intra-encoder dropout if n_encoders > 1
            'tied_emb': False,          # Share embeddings: (False|2way|3way)
            'direction': None,          # Network directionality, i.e. en->de
            'max_len': 80,              # Reject sentences where 'bucket_by' length > 80
            'bucket_by': None,          # A key like 'en' to define w.r.t which dataset
                                        # the batches will be sorted
            'bucket_order': None,       # Curriculum: ascending/descending/None
            'sampler_type': 'bucket',   # bucket or approximate
            'sched_sampling': 0,        # Scheduled sampling ratio
            'short_list': 0,            # Short list vocabularies (0: disabled)
            'bos_type': 'emb',          # 'emb': default learned emb
            'bos_activ': None,          #
            'bos_dim': None,            #
            'out_logic': 'simple',      # 'simple' or 'deep' output
            'dec_inp_activ': None,      # Non-linearity for GRU2 input in dec
        }

    def __init__(self, opts, beat_platform=False):
        super().__init__()

        # Are we operating in the BEAT platform?
        self.beat_platform = beat_platform

        # opts -> config file sections {.model, .data, .vocabulary, .train}
        self.opts = opts

        # Vocabulary objects
        self.vocabs = {}

        # Each auxiliary loss should be stored inside this dictionary
        # in order to be taken into account by the mainloop for multi-tasking
        self.aux_loss = {}

        # Setup options
        self.opts.model = self.set_model_options(opts.model)

        # Parse topology & languages
        self.topology = Topology(self.opts.model['direction'])

        # Load vocabularies here: if beat_platform, content=vocab in an OrderedDict, otherwise content=filename
        for name, content in self.opts.vocabulary.items():
            self.vocabs[name] = Vocabulary(content, short_list=self.opts.model['short_list'], beat_platform=self.beat_platform)

        # Inherently non multi-lingual aware
        slangs = self.topology.get_src_langs()
        tlangs = self.topology.get_trg_langs()
        if slangs:
            self.sl = slangs[0]
            self.src_vocab = self.vocabs[self.sl]
            self.n_src_vocab = len(self.src_vocab)
        if tlangs:
            self.tl = tlangs[0]
            self.trg_vocab = self.vocabs[self.tl]
            self.n_trg_vocab = len(self.trg_vocab)
            # Need to be set for early-stop evaluation
            # NOTE: This should come from config or elsewhere
            self.val_refs = self.opts.data['val_set'][self.tl]

        # Check vocabulary sizes for 3way tying
        if self.opts.model.get('tied_emb', False) not in [False, '2way', '3way']:
            raise RuntimeError(
                "'{}' not recognized for tied_emb.".format(self.opts.model['tied_emb']))

        if self.opts.model.get('tied_emb', False) == '3way':
            assert self.n_src_vocab == self.n_trg_vocab, \
                "The vocabulary sizes do not match for 3way tied embeddings."

    def __repr__(self):
        s = super().__repr__() + '\n'
        for vocab in self.vocabs.values():
            s += "{}\n".format(vocab)
        s += "{}\n".format(get_n_params(self))
        return s

    def set_model_options(self, model_opts):
        self.set_defaults()
        for opt, value in model_opts.items():
            if opt in self.defaults:
                # Override defaults from config
                self.defaults[opt] = value
            else:
                logger.info('Warning: unused model option: {}'.format(opt))
        return self.defaults

    def reset_parameters(self):
        for name, param in self.named_parameters():
            # Skip 1-d biases and scalars
            if param.requires_grad and param.dim() > 1:
                nn.init.kaiming_normal_(param.data)
        # Reset padding embedding to 0
        if hasattr(self, 'enc') and hasattr(self.enc, 'emb'):
            with torch.no_grad():
                self.enc.emb.weight.data[0].fill_(0)

    def setup(self, is_train=True):
        """Sets up NN topology by creating the layers."""
        ########################
        # Create Textual Encoder
        ########################
        self.enc = TextEncoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['enc_dim'],
            n_vocab=self.n_src_vocab,
            bidirectional=self.opts.model['enc_bidirectional'],
            rnn_type=self.opts.model['enc_type'],
            dropout_emb=self.opts.model['dropout_emb'],
            dropout_ctx=self.opts.model['dropout_ctx'],
            dropout_rnn=self.opts.model['dropout_enc'],
            num_layers=self.opts.model['n_encoders'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            layer_norm=self.opts.model['enc_lnorm'])

        self.ctx_sizes = {str(self.sl): self.enc.ctx_size}

        ################
        # Create Decoder
        ################
        Decoder = get_decoder(self.opts.model['dec_variant'])
        self.dec = Decoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name=str(self.sl),
            tied_emb=self.opts.model['tied_emb'],
            dec_init=self.opts.model['dec_init'],
            dec_init_size=self.opts.model['dec_init_size'],
            dec_init_activ=self.opts.model['dec_init_activ'],
            att_type=self.opts.model['att_type'],
            att_temp=self.opts.model['att_temp'],
            att_activ=self.opts.model['att_activ'],
            att_ctx2hid=self.opts.model['att_ctx2hid'],
            transform_ctx=self.opts.model['att_transform_ctx'],
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout_out'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            sched_sample=self.opts.model['sched_sampling'],
            bos_type=self.opts.model['bos_type'],
            bos_dim=self.opts.model['bos_dim'],
            bos_activ=self.opts.model['bos_activ'],
            bos_bias=self.opts.model['bos_type'] == 'feats',
            out_logic=self.opts.model['out_logic'],
            dec_inp_activ=self.opts.model['dec_inp_activ'])

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            self.enc.emb.weight = self.dec.emb.weight

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
            beat_platform=self.beat_platform)
        logger.info(self.dataset)
        return self.dataset

    def get_bos(self, batch_size):
        """Returns a representation for <bos> embeddings for decoding."""
        return torch.LongTensor(batch_size).fill_(self.trg_vocab['<bos>'])

    def encode(self, batch, **kwargs):
        """Encodes all inputs and returns a dictionary.

        Arguments:
            batch (dict): A batch of samples with keys designating the
                information sources.

        Returns:
            dict:
                A dictionary where keys are source modalities compatible
                with the data loader and the values are tuples where the
                elements are encodings and masks. The mask can be ``None``
                if the relevant modality does not require a mask.
        """
        d = {str(self.sl): self.enc(batch[self.sl])}
        if 'feats' in batch:
            d['feats'] = (batch['feats'], None)
        return d

    def forward(self, batch, **kwargs):
        """Computes the forward-pass of the network and returns batch loss.

        Arguments:
            batch (dict): A batch of samples with keys designating the source
                and target modalities.

        Returns:
            Tensor:
                A scalar loss normalized w.r.t batch size and token counts.
        """
        # Get loss dict
        enc = self.encode(batch)

        #result = self.dec(self.encode(batch), batch[self.tl])
        result = self.dec(enc, batch[self.tl])
        result['n_items'] = torch.nonzero(batch[self.tl][1:]).shape[0]
        return result

    def test_performance(self, data_loader, dump_file=None):
        """Computes test set loss over the given DataLoader instance."""
        loss = Loss()
        for batch in pbar(data_loader, unit='batch'):
            batch.device(DeviceManager.DEVICE)
            out = self.forward(batch)
            loss.update(out['loss'], out['n_items'])

        return [
            Metric('LOSS', loss.get(), higher_better=False),
        ]

    def get_decoder(self, task_id=None):
        """Compatibility function for multi-tasking architectures."""
        return self.dec

    def register_tensorboard(self, handle):
        """Stores tensorboard hook for custom logging."""
        self.tboard = handle

    @staticmethod
    def beam_search(models, data_loader, task_id=None, beam_size=12, max_len=200,
                    lp_alpha=0., suppress_unk=False, n_best=False):
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
            max_len, max_batch_size, k, dtype=torch.long, device=DeviceManager.DEVICE)
        mask = torch.arange(max_batch_size * k, device=DeviceManager.DEVICE)
        nll_storage = torch.zeros(max_batch_size, device=DeviceManager.DEVICE)

        for batch in pbar(data_loader, unit='batch'):
            batch.device(DeviceManager.DEVICE)

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

            # Sanity check one of the context dictionaries for dimensions
            check_context_ndims(ctx_dicts[0])

            # Get initial decoder state (N*H)
            h_ts = [f_init(ctx_dict) for f_init, ctx_dict in zip(f_inits, ctx_dicts)]

            # we always have <bos> tokens except that the returned embeddings
            # may differ from one model to another.
            idxs = models[0].get_bos(batch.size).to(DeviceManager.DEVICE)

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
                results.extend(vocab.list_of_idxs_to_sents(hyps.tolist()))

        # Recover order of the samples if necessary
        return sort_predictions(data_loader, results)
