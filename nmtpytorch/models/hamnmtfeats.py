# -*- coding: utf-8 -*-
from collections import OrderedDict
import math
import logging

import torch
from torch import nn

from ..datasets import MultimodalDataset
from ..layers import ConditionalMMDecoder, FF, MultimodalTextEncoder
from .nmt import NMT

logger = logging.getLogger('nmtpytorch')


class HybridAttentiveMNMTFeatures(NMT):
    """An end-to-end sequence-to-sequence NMT model with visual attention over
    pre-extracted convolutional features.
    """
    def set_defaults(self):
        # Set parent defaults
        super().set_defaults()
        self.defaults.update({
            'fusion_type': 'concat',    # Multimodal context fusion (sum|mul|concat)
            'fusion_activ': 'tanh',     # Multimodal context non-linearity
            'n_channels': 2048,         # depends on the features used
            'mm_att_type': 'md-dd',     # multimodal attention type
                                        # md: modality dep.
                                        # mi: modality indep.
                                        # dd: decoder state dep.
                                        # di: decoder state indep.
            'out_logic': 'deep',        # simple vs deep output
            'persistent_dump': False,   # To save activations during beam-search
        })

    def __init__(self, opts):
        super().__init__(opts)

    def setup(self, is_train=True):
        # Textual context dim
        txt_ctx_size = self.ctx_sizes[self.sl]

        # Add vis ctx size
        self.ctx_sizes['image'] = txt_ctx_size

        # Init decoder from the projected feats
        self.opts.model['dec_init'] = 'feats'
        self.opts.model['dec_init_size'] = txt_ctx_size

        # Add visual context transformation (sect. 3.2 in paper)
        self.ff_img = FF(self.opts.model['n_channels'], txt_ctx_size)

        self.enc = MultimodalTextEncoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['enc_dim'],
            n_vocab=self.n_src_vocab,
            rnn_type=self.opts.model['enc_type'],
            dropout_emb=self.opts.model['dropout_emb'],
            dropout_ctx=self.opts.model['dropout_ctx'],
            dropout_rnn=self.opts.model['dropout_enc'],
            num_layers=self.opts.model['n_encoders'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            layer_norm=self.opts.model['enc_lnorm'],
            feat_size=txt_ctx_size,
            feat_activ='tanh',
            feat_fusion='encdecinit')

        # Create Decoder
        self.dec = ConditionalMMDecoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name=str(self.sl),
            fusion_type=self.opts.model['fusion_type'],
            fusion_activ=self.opts.model['fusion_activ'],
            tied_emb=self.opts.model['tied_emb'],
            dec_init=self.opts.model['dec_init'],
            dec_init_size=txt_ctx_size,
            dec_init_activ='tanh',
            att_type=self.opts.model['att_type'],
            mm_att_type=self.opts.model['mm_att_type'],
            out_logic=self.opts.model['out_logic'],
            att_activ=self.opts.model['att_activ'],
            transform_ctx=self.opts.model['att_transform_ctx'],
            att_ctx2hid=False,
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout_out'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            persistent_dump=self.opts.model['persistent_dump'])

        # Tie initializers
        self.enc.ff_vis.weight = self.dec.ff_dec_init.weight

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            self.enc.emb.weight = self.dec.emb.weight

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        dataset = MultimodalDataset(
            data=self.opts.data[split + '_set'],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model.get('max_len', None),
            order_file=self.opts.data[split + '_set'].get('ord', None))
        logger.info(dataset)
        return dataset

    def encode(self, batch, **kwargs):
        # Transform the channel dim to context dim: k*k*c
        feats = self.ff_img(batch['image'])

        # Pool to global features
        pooled_feats = feats.mean(0, keepdim=True)

        # Get source language encodings (S*B*C)
        text_encoding = self.enc(batch[self.sl], v=pooled_feats)

        return {
            str(self.sl): text_encoding,
            'image': (feats, None),
            'feats': (pooled_feats, None),
        }
