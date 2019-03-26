# -*- coding: utf-8 -*-
import random
import logging

import torch

from ..datasets import MultimodalDataset
from ..layers import ConditionalMMDecoder, TextEncoder, FF
from ..utils.topology import Topology
from .nmt import NMT

logger = logging.getLogger('nmtpytorch')


class AttentiveMNMTFeaturesColingMasked(NMT):
    """An end-to-end sequence-to-sequence NMT model with visual attention over
    pre-extracted convolutional features.
    """
    def set_defaults(self):
        # Set parent defaults
        super().set_defaults()
        self.defaults.update({
            'alpha_c': 0.0,             # doubly stoch. attention
            'fusion_type': 'concat',    # Multimodal context fusion (sum|mul|concat)
            'fusion_activ': 'tanh',     # Multimodal context non-linearity
            'n_channels': 2048,         # depends on the features used
            'mm_att_type': 'md-dd',     # multimodal attention type
                                        # md: modality dep.
                                        # mi: modality indep.
                                        # dd: decoder state dep.
                                        # di: decoder state indep.
            'out_logic': 'deep',        # simple vs deep output
            'test_direction': None,     # Test-time modalities can differ (optional)
            'p_mask': 0.5               # Masking probability
        })

    def __init__(self, opts):
        super().__init__(opts)
        if self.opts.model['alpha_c'] > 0:
            self.aux_loss['alpha_reg'] = 0.0

        self.test_topology = self.topology
        if 'test_direction' in self.opts.model:
            self.test_topology = Topology(self.opts.model['test_direction'])

    def setup(self, is_train=True):
        txt_ctx_size = list(self.ctx_sizes.values())[0]

        # Add visual context transformation (sect. 3.2 in paper)
        self.ff_img = FF(self.opts.model['n_channels'], txt_ctx_size)

        # Add vis ctx size
        self.ctx_sizes['image'] = txt_ctx_size

        ########################
        # Create Textual Encoder
        ########################
        self.enc = TextEncoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['enc_dim'],
            n_vocab=self.n_src_vocab,
            rnn_type=self.opts.model['enc_type'],
            dropout_emb=self.opts.model['dropout_emb'],
            dropout_ctx=self.opts.model['dropout_ctx'],
            dropout_rnn=self.opts.model['dropout_enc'],
            num_layers=self.opts.model['n_encoders'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'])

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
            emb_gradscale=self.opts.model['emb_gradscale'])

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            self.enc.emb.weight = self.dec.emb.weight

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        topo = self.topology if mode == 'train' else self.test_topology
        dataset = MultimodalDataset(
            data=self.opts.data[split + '_set'],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=topo,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model.get('max_len', None),
            order_file=self.opts.data[split + '_set'].get('ord', None))
        logger.info(dataset)
        return dataset

    def encode(self, batch, **kwargs):
        # Let's start with a None mask by assuming that
        # we have a fixed-length feature collection
        feats, feats_mask = self.ff_img(batch['image']), None
        src_lang = str(self.sl)
        # Randomly mask entities with p=p_mask
        if self.training and random.random() >= (1 - self.opts.model['p_mask']):
            src_lang += '_masked'

        return {
            'image': (feats, feats_mask),
            str(self.sl): self.enc(batch[src_lang]),
        }

    def forward(self, batch, **kwargs):
        result = super().forward(batch)

        if self.training and self.opts.model['alpha_c'] > 0:
            alpha_loss = (
                1 - torch.cat(self.dec.history['alpha_img']).sum(0)).pow(2).sum(0)
            self.aux_loss['alpha_reg'] = alpha_loss.mean().mul(
                self.opts.model['alpha_c'])

        return result
