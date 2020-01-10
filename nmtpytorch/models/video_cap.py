# -*- coding: utf-8 -*-
from collections import OrderedDict
import logging

from torch import nn

from ..vocabulary import Vocabulary
from ..layers.decoders import get_decoder
from ..utils.topology import Topology
from ..datasets import MultimodalDataset
from ..layers import FF

from . import NMT

logger = logging.getLogger('nmtpytorch')


class VideoCap(NMT):
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            'n_obj_labels': 601,        # # of object labels available
            'n_act_labels': 339,        # # of action labels available
            'feat_dim': 128,            # Projected feature dim
            'feat_activ': None,         # feature activation type
            'feat_bias': True,          # Bias for the label embeddings
            'feat_fusion': None,        # concat or last
            'emb_dim': 128,             # Target embedding sizes
            'enc_dim': 256,             # Encoder hidden size
            'encode': False,            # If `True`, add a GRU on top
            'encode_bidir': False,      # Bidirectional?
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
            'dropout_feats': 0,         # Simple dropout to features
            'dropout_out': 0,           # Simple dropout to decoder output
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

        tlangs = self.topology.get_trg_langs()
        self.tl = tlangs[0]
        self.trg_vocab = self.vocabs[self.tl]
        self.n_trg_vocab = len(self.trg_vocab)
        self.val_refs = self.opts.data['val_set'][self.tl]

        # Video context
        self.sl = 'vid'
        self.ctx_sizes = {self.sl: self.opts.model['feat_dim']}

    def setup(self, is_train=True):
        """Sets up NN topology by creating the layers."""

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
            sched_sample=self.opts.model['sched_sampling'],
            bos_type=self.opts.model['bos_type'],
            bos_dim=self.opts.model['bos_dim'],
            bos_activ=self.opts.model['bos_activ'],
            bos_bias=self.opts.model['bos_type'] == 'feats',
            out_logic=self.opts.model['out_logic'],
            dec_inp_activ=self.opts.model['dec_inp_activ'])

        ################
        # Video features
        ################
        obj_layers = OrderedDict([
            ('proj1', FF(self.opts.model['n_obj_labels'], self.opts.model['feat_dim'], bias=self.opts.model['feat_bias'])),
            #('proj2', FF(self.opts.model['feat_dim'], self.opts.model['feat_dim'], bias=self.opts.model['feat_bias'])),
        ])
        self.video_fn = nn.Sequential(obj_layers)

        self.dropout = nn.Dropout(self.opts.model['dropout_feats'])

#         self.act_emb = None
        # if self.opts.model['feat_fusion'] is not None:
            # # Assume action features are available then
            # self.act_emb = FF(self.opts.model['n_act_labels'],
                              # self.opts.model['feat_dim'])
            # if self.opts.model['feat_fusion'] == 'last':
                # # Last concat
                # self.fuse_op = lambda obj, act: torch.cat((obj, act))
            # elif self.opts.model['feat_fusion'] == 'concat':
                # self.concat_proj = FF(
                    # 2 * self.opts.model['feat_dim'],
                    # self.opts.model['feat_dim'],
                    # bias=self.opts.model['feat_bias'],
                    # activ=self.opts.model['feat_activ'])
                # self.fuse_op = lambda obj, act: self.concat_proj(torch.cat(
                    # (obj, act.expand(obj.shape[0], -1, -1)), dim=-1))

        if self.opts.model['encode']:
            self.enc = nn.GRU(
                self.opts.model['feat_dim'], self.opts.model['feat_dim'],
                n_layers=self.opts.model['n_encoders'],
                bidirectional=self.opts.model['encode_bidir'],
                batch_first=False)
        else:
            # no-op
            self.enc = lambda x: (x, x)

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        order_file = self.opts.data[split + '_set'].get(
            'keys.beam' if mode == 'beam' else 'keys', None)
        self.dataset = MultimodalDataset(
            data=self.opts.data['{}_set'.format(split)],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model['max_len'],
            bucket_order=self.opts.model['bucket_order'],
            sampler_type=self.opts.model['sampler_type'],
            order_file=order_file)
        logger.info(self.dataset)
        return self.dataset

    def encode(self, batch, **kwargs):
        feats = self.dropout(self.video_fn(batch['obj']))
        out, _ = self.enc(feats)
        return {str(self.sl): (out, None)}
