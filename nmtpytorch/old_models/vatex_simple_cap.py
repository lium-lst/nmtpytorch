# -*- coding: utf-8 -*-
from ..logger import Logger

from torch import nn

from ..layers import FF, Pool
from ..layers.decoders import VectorDecoder
from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from ..datasets import MultimodalDataset
from . import NMT

log = Logger()


class VATEXSimpleCaptioner(NMT):
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            'dropout': 0,               # Simple dropout to source embeddings
            'dec_dim': 256,             # Decoder hidden size
            'emb_dim': 200,             # Decoder embedding size
            'dec_init': 'feats',        # How to initialize decoder (zero/mean_ctx/feats)
            'feat_dim': 4383,           # Label vocab size
            'feat_proj_dim': 512,       # Intermediate projection
            'feat_proj_activ': None,    # Non-linearity
            'feat_aggregate': None,     # For I3D initialized models
            'dec_init_activ': 'tanh',   # Non-linearity
            'direction': None,          # Network directionality, i.e. en->de
            'max_len': None,            # Reject sentences where 'bucket_by' length > 80
            'bucket_by': None,          # A key like 'en' to define w.r.t which dataset
                                        # the batches will be sorted
            'bucket_order': None,       # Curriculum: ascending/descending/None
            'sampler_type': 'random',   # bucket or approximate
            'sched_sampling': 0,        # Scheduled sampling ratio
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
        self.src = self.topology.first_src
        self.tl = self.topology.first_trg

        # Load vocabularies here
        for name, fname in self.opts.vocabulary.items():
            self.vocabs[name] = Vocabulary(fname, name=name)

        self.tl = list(self.vocabs.keys())[0]
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
        log.log(self.dataset)
        return self.dataset

    def setup(self, is_train=True):
        if self.opts.model['dec_init'] == 'zero':
            # No visual input at all
            self.enc = lambda x: x
        else:
            layers = [FF(
                self.opts.model['feat_dim'],
                self.opts.model['feat_proj_dim'],
                activ=self.opts.model['feat_proj_activ'])]
            if self.opts.model['feat_aggregate']:
                layers.append(
                    Pool(self.opts.model['feat_aggregate'], pool_dim=0))
            self.enc = nn.Sequential(*layers)
        ################
        # Create Decoder
        ################
        self.dec = VectorDecoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            rnn_type='gru',
            dec_init=self.opts.model['dec_init'],
            dec_init_activ=self.opts.model['dec_init_activ'],
            dec_init_size=self.opts.model['feat_proj_dim'],
            ctx_name='feats',
            ctx_size_dict={'feats': 500},     # dummy, not used
            n_vocab=self.n_trg_vocab,
            dropout_out=self.opts.model['dropout'])

    def encode(self, batch, **kwargs):
        # Project
        feats = self.enc(batch[self.src])
        return {
            'feats': (feats, None),
        }

    def prepare_outputs(self, beam_results):
        json_results = []
        for idx, cap in enumerate(beam_results):
            json_results.append({
                'image_id': self.dataset.datasets[self.src].keys[idx],
                'caption': cap})
        return json_results
