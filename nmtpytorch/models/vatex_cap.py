# -*- coding: utf-8 -*-
import logging

from torch import nn

from ..layers import FF, ArgSelect
from ..layers.decoders import ConditionalDecoder
from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from ..datasets import MultimodalDataset
from . import NMT

logger = logging.getLogger('nmtpytorch')


class VATEXCaptioner(NMT):
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            'feat_dim': 1024,           # Source feature dim
            'feat_proj_dim': 128,       # FF-layer dim
            'feat_proj_activ': None,    # Linear FF by default
            'enc_type': None,           # None, gru or lstm
            'enc_dim': 128,             # Hidden dim of the encoder
            'n_encoders': 1,            # Only used if enc_type != None
            'enc_bidir': False,         # Bi-directional?
            'dropout_feat': 0,          # Simple dropout to source embeddings
            'dec_dim': 256,             # Decoder hidden size
            'emb_dim': 256,             # Decoder embedding size
            'dec_init': 'mean_ctx',     # How to initialize decoder (zero/mean_ctx/feats)
            'dropout_out': 0,           # Simple dropout to decoder output
            'direction': None,          # Network directionality, i.e. en->de
            'max_len': None,            # Reject sentences where 'bucket_by' length > 80
            'bucket_by': None,          # A key like 'en' to define w.r.t which dataset
                                        # the batches will be sorted
            'bucket_order': None,       # Curriculum: ascending/descending/None
            'sampler_type': 'bucket',   # bucket or approximate
            'sched_sampling': 0,        # Scheduled sampling ratio
            'out_logic': 'simple',      # 'simple' or 'deep' output
        }

    def __init__(self, opts):
        # Don't call NMT init as it's too different from this model
        nn.Module.__init__(self)

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

        # Load vocabularies here
        for name, fname in self.opts.vocabulary.items():
            self.vocabs[name] = Vocabulary(fname, name=name)

        # NOTE: Hardcode 'en' key as target modality
        self.tl = 'en'
        self.trg_vocab = self.vocabs[self.tl]
        self.n_trg_vocab = len(self.trg_vocab)
        self.val_refs = self.opts.data['val_set'][self.tl]

    def reset_parameters(self):
        for name, param in self.named_parameters():
            # Skip 1-d biases and scalars
            if param.requires_grad and param.dim() > 1:
                nn.init.kaiming_normal_(param.data)

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
            repeat_by=10 if mode != 'beam' else 1)
        logger.info(self.dataset)
        return self.dataset

    def setup(self, is_train=True):
        ##############################
        # Copied from label_classifier
        ##############################
        feat_proj = FF(self.opts.model['feat_dim'],
                       self.opts.model['feat_proj_dim'],
                       activ=self.opts.model['feat_proj_activ'],
                       bias=False)

        layers = [feat_proj]
        out_dim = self.opts.model['feat_proj_dim']

        if self.opts.model['enc_type'] is not None:
            RNN = getattr(nn, self.opts.model['enc_type'].upper())
            enc = RNN(self.opts.model['feat_proj_dim'],
                      self.opts.model['enc_dim'],
                      self.opts.model['n_encoders'],
                      bidirectional=self.opts.model['enc_bidir'],
                      batch_first=False)
            out_dim = self.opts.model['enc_dim']
            layers.append(enc)
            layers.append(ArgSelect(0))
            if self.opts.model['enc_bidir']:
                layers.append(FF(out_dim * 2, out_dim, bias=False))

        self.ctx_dim = out_dim

        if self.opts.model['dropout_feat'] > 0:
            layers.append(nn.Dropout(self.opts.model['dropout_feat']))

        # Construct one encoder abstraction
        self.enc = nn.Sequential(*layers)

        ################
        # Create Decoder
        ################
        self.dec = ConditionalDecoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            rnn_type='gru',
            ctx_name='i3d',
            ctx_size_dict={'i3d': self.ctx_dim},
            n_vocab=self.n_trg_vocab,
            dropout_out=self.opts.model['dropout_out'])

    def encode(self, batch, **kwargs):
        return {
            'i3d': (self.enc(batch['i3d']), None),
        }

    def prepare_outputs(self, beam_results):
        json_results = []
        for idx, cap in enumerate(beam_results):
            json_results.append({
                'image_id': self.dataset.datasets['i3d'].keys[idx],
                'caption': cap})
        return json_results
