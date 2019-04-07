# -*- coding: utf-8 -*-
from collections import OrderedDict
import math
import logging

import torch
from torch import nn

from ..datasets import MultimodalDataset
from ..layers import ConditionalMMDecoder, TextEncoder, FF
from .nmt import NMT

logger = logging.getLogger('nmtpytorch')


class AttentiveMNMTFeaturesColing(NMT):
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
            'vis_activ': 'linear',      # Visual feature transformation activ.
            'n_channels': 2048,         # depends on the features used
            'mm_att_type': 'md-dd',     # multimodal attention type
                                        # md: modality dep.
                                        # mi: modality indep.
                                        # dd: decoder state dep.
                                        # di: decoder state indep.
            'out_logic': 'deep',        # simple vs deep output
            'persistent_dump': False,   # To save activations during beam-search
            'preatt': False,            # Apply filtered attention
            'preatt_activ': 'ReLU',     # Activation for convatt block
        })

    def __init__(self, opts):
        super().__init__(opts)
        if self.opts.model['alpha_c'] > 0:
            self.aux_loss['alpha_reg'] = 0.0

    def setup(self, is_train=True):
        # Textual context dim
        txt_ctx_size = self.ctx_sizes[self.sl]

        # Add visual context transformation (sect. 3.2 in paper)
        self.ff_img = FF(
            self.opts.model['n_channels'], txt_ctx_size,
            activ=self.opts.model['vis_activ'])

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
            emb_gradscale=self.opts.model['emb_gradscale'],
            persistent_dump=self.opts.model['persistent_dump'])

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            self.enc.emb.weight = self.dec.emb.weight

        if self.opts.model['preatt']:
            # From 640+640 to 640
            # To 1*W*W for attention scores
            in_channels = sum(self.ctx_sizes.values())
            self.preatt = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(in_channels, self.ctx_sizes[self.sl], 1, 1)),
                ('nlin1', getattr(nn, self.opts.model['preatt_activ'])()),
                ('conv2', nn.Conv2d(self.ctx_sizes[self.sl], 1, 1, 1)),
                ('nlin2', getattr(nn, self.opts.model['preatt_activ'])()),
            ]))

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
        # Transform the features to context dim
        feats = self.ff_img(batch['image'])

        # Get source language encodings (S*B*C)
        text_encoding = self.enc(batch[self.sl])

        if self.opts.model['preatt']:
            # Infer spatial width/height
            w = int(math.sqrt(feats.shape[0]))

            # image features will come as: (HW,B,C) -> (B, C, HW) -> (B, C, H, W)
            conv_map = feats.permute(1, 2, 0).view(
                *feats.shape[1:], w, w)

            # Get last encoding (B*C)
            last_encoding = text_encoding[0][-1]

            # Tile over spatial dimensions: B*C*H*W
            tiled_encoding = last_encoding[..., None, None].expand(
                -1, -1, *conv_map.shape[2:])

            # Final concat representation: B*(D+C)*H*W
            concat = torch.cat([conv_map, tiled_encoding], dim=1)

            att_scores = self.preatt(concat)
            self.pre_att = nn.functional.softmax(
                att_scores.view(batch.size, -1), dim=1).view_as(att_scores)

            # Filter features
            feats = conv_map * self.pre_att

            # Get features into (B, C, HW) and then (HW, B, C)
            feats = feats.view(*feats.shape[:2], -1).permute(2, 0, 1)

        return {
            str(self.sl): text_encoding,
            'image': (feats, None),
        }

    def forward(self, batch, **kwargs):
        result = super().forward(batch)

        if self.training and self.opts.model['alpha_c'] > 0:
            alpha_loss = (
                1 - torch.cat(self.dec.history['alpha_img']).sum(0)).pow(2).sum(0)
            self.aux_loss['alpha_reg'] = alpha_loss.mean().mul(
                self.opts.model['alpha_c'])

        return result
