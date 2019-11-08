# -*- coding: utf-8 -*-
from collections import OrderedDict
import logging

import torch
from torch import nn

from ..layers import FF
from ..layers.decoders import get_decoder
from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from ..datasets import MultimodalDataset
from . import NMT

logger = logging.getLogger('nmtpytorch')


class VideoCaptioner(NMT):
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            'n_labels': 601,            # # of input labels available
            #'n_act_labels': 339,        # # of input action labels
            'feat_dim': 128,            # Projected feature dim
            'feat_activ': None,         # feature activation type
            'feat_bias': True,          # Bias for the label embeddings
            'feat_fusion': None,        # concat or last
            'encoder': False,           # Add an encoder or not
            'emb_dim': 128,             # Target embedding sizes
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
            'dropout_out': 0,           # Simple dropout to decoder output
            'direction': None,          # Network directionality, i.e. en->de
            'max_len': None,            # Reject sentences where 'bucket_by' length > 80
            'bucket_by': None,          # A key like 'en' to define w.r.t which dataset
                                        # the batches will be sorted
            'bucket_order': None,       # Curriculum: ascending/descending/None
            'master': None,             # Coordinating dataset (should be the JSON one)
            'sampler_type': 'bucket',   # bucket or approximate
            'sched_sampling': 0,        # Scheduled sampling ratio
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

        # NOTE: Hardcode 'en' key as target modality
        self.tl = 'en'
        self.trg_vocab = self.vocabs[self.tl]
        self.n_trg_vocab = len(self.trg_vocab)
        self.val_refs = self.opts.data['val_set'][self.tl]

        self.ctx_sizes = {'obj': self.opts.model['feat_dim']}

    def reset_parameters(self):
        for name, param in self.named_parameters():
            # Skip 1-d biases and scalars
            if param.requires_grad and param.dim() > 1:
                nn.init.kaiming_normal_(param.data)

    def setup(self, is_train=True):
        """Sets up NN topology by creating the layers."""
        # Actual embeddings
        emb = FF(self.opts.model['n_labels'], self.opts.model['feat_dim'])

        # Add another projection layer
        proj = FF(
            self.opts.model['feat_dim'], self.opts.model['feat_dim'])

        self.emb = nn.Sequential(OrderedDict([
            ('emb', emb),
            ('emb_proj', proj),
        ]))

        self.act_emb = None
        if self.opts.model['feat_fusion'] is not None:
            # Assume action features are available then
            self.act_emb = FF(self.opts.model['n_act_labels'],
                              self.opts.model['feat_dim'])
            if self.opts.model['feat_fusion'] == 'last':
                # Last concat
                self.fuse_op = lambda obj, act: torch.cat((obj, act))
            elif self.opts.model['feat_fusion'] == 'concat':
                self.concat_proj = FF(
                    2 * self.opts.model['feat_dim'],
                    self.opts.model['feat_dim'],
                    bias=self.opts.model['feat_bias'],
                    activ=self.opts.model['feat_activ'])
                self.fuse_op = lambda obj, act: self.concat_proj(torch.cat(
                    (obj, act.expand(obj.shape[0], -1, -1)), dim=-1))

        self.dropout = nn.Dropout(self.opts.model['dropout_emb'])

        if self.opts.model['encoder']:
            self.enc = nn.GRU(
                self.opts.model['feat_dim'], self.opts.model['feat_dim'],
                1, bias=True, batch_first=False, bidirectional=False)

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
            ctx_name='obj',
            tied_emb=False,
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

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        dataset = MultimodalDataset(
            data=self.opts.data['{}_set'.format(split)],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model['max_len'],
            bucket_order=self.opts.model['bucket_order'],
            sampler_type=self.opts.model['sampler_type'],
            master=self.opts.model['master'])
        logger.info(dataset)
        return dataset

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
        feat = self.emb(batch['obj'])
        if self.act_emb is not None:
            act = self.act_emb(batch['act'])
            feat = self.fuse_op(feat, act)

        # Apply dropout
        feat = self.dropout(feat)

        if self.opts.model['encoder']:
            feat, _ = self.enc(feat)

        return {
            'obj': (feat, None),
        }
