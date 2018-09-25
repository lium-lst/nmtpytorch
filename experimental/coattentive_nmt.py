# -*- coding: utf-8 -*-
import logging

import torch

from ..layers import (TextEncoder, CoAttention, MultiSourceConditionalDecoder,
                      SequenceConvolution, MultiHeadCoAttention)
from ..datasets import MultimodalDataset
from .nmt import NMT

logger = logging.getLogger('nmtpytorch')


class CoAttentiveMNMTFeatures(NMT):
    """NMT model doing co-attention with addtional sequence of features."""
    def set_defaults(self):
        # Set parent defaults
        super().set_defaults()
        self.defaults.update({
            'fusion_type': 'hierarchical',    # Multimodal context fusion (sum|mul|concat)
            'n_channels': 2048,         # depends on the features used
            'alpha_c': 0.0,             # doubly stoch. attention
            'img_sequence': False,      # if true img is sequence of img features,
                                        # otherwise it's a conv map
            'coattention': 'to_text',   # direction of the coattention (to_text|to_img)
            'include_txt': True,
            'include_img': True,
            'include_img2txt': True,
            'include_txt2img': True,
            'txt_conv_filters': [0, 128, 128, 128, 128],
            'img_conv_filters': [0, 128, 128, 128, 128],
            'txt_maxpool': 5,
            'img_maxpool': 5,
            'co_att_type': 'mlp',  # (mlp|multihead)
            'co_att_heads': 8,
        })

    def __init__(self, opts):
        super().__init__(opts)
        if self.opts.model['alpha_c'] > 0:
            self.aux_loss['alpha_reg'] = 0.0

        if not self.opts.model['coattention'] in ['to_text', 'to_img']:
            raise ValueError("Unknown type of co-attention: {}.".format(
                self.opts.model['coattention']))

    def setup(self, is_train=True):
        self.ctx_sizes['image'] = self.opts.model['n_channels']

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

        self.ctx_names = []

        if self.opts.model['include_txt']:
            self.ctx_names.append(str(self.sl))
        if self.opts.model['include_img']:
            self.ctx_names.append('image')

        if (self.opts.model['include_img2txt'] or
                self.opts.model['include_txt2img']):
            txt_dim = sum(self.opts.model['txt_conv_filters'])
            img_dim = sum(self.opts.model['img_conv_filters'])

            self.img_conv = SequenceConvolution(
                input_dim=self.opts.model['n_channels'],
                filters=self.opts.model['img_conv_filters'],
                max_pool_stride=self.opts.model['img_maxpool'])

            self.txt_conv = SequenceConvolution(
                input_dim=self.opts.model['enc_dim'] * 2,
                filters=self.opts.model['txt_conv_filters'],
                max_pool_stride=self.opts.model['txt_maxpool'])

            if self.opts.model['co_att_type'] == 'mlp':
                self.coattention = CoAttention(
                    ctx_1_dim=txt_dim,
                    ctx_2_dim=img_dim,
                    bottleneck=min(txt_dim, img_dim))
                self.ctx_sizes['img2txt'] = img_dim
                self.ctx_sizes['txt2img'] = txt_dim
            elif self.opts.model['co_att_type'] == 'multihead':
                att_dim = min(txt_dim, img_dim)
                self.coattention = MultiHeadCoAttention(
                    ctx_1_dim=txt_dim,
                    ctx_2_dim=img_dim,
                    bottleneck=att_dim,
                    head_count=self.opts.model['co_att_heads'])
                self.ctx_sizes['img2txt'] = att_dim
                self.ctx_sizes['txt2img'] = att_dim
            else:
                raise ValueError("Unknown coattentino type: {}".format(
                    self.opts.model['co_att_type']))

            if self.opts.model['include_img2txt']:
                self.ctx_names.append('img2txt')

            if self.opts.model['include_txt2img']:
                self.ctx_names.append('txt2img')

        # Create Decoder
        self.dec = MultiSourceConditionalDecoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name=str(self.sl),
            ctx_names=self.ctx_names,
            fusion_type=self.opts.model['fusion_type'],
            tied_emb=self.opts.model['tied_emb'],
            dec_init=self.opts.model['dec_init'],
            att_type=self.opts.model['att_type'],
            att_activ=self.opts.model['att_activ'],
            transform_ctx=self.opts.model['att_transform_ctx'],
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
        dataset = MultimodalDataset(
            data=self.opts.data[split + '_set'],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model.get('max_len', None))
        logger.info(dataset)
        return dataset

    def encode(self, batch, **kwargs):
        # Get features into (n,c,w*h) and then (w*h,n,c)
        # Let's start with a None mask by assuming that
        # we have a fixed-length feature collection
        feats_mask = None

        # Be it Numpy or NumpySequence, they return
        # (n_samples, feat_dim, t) by default
        # Convert it to (t, n_samples, feat_dim)
        feats = batch['image'].view(
            (*batch['image'].shape[:2], -1)).permute(2, 0, 1)

        if self.opts.model['img_sequence']:
            # Let's create mask in this case
            feats_mask = feats.ne(0).float().sum(2).ne(0).float()

        txt_encoded, txt_mask = self.enc(batch[self.sl])

        ctx_dict = {}
        ctx_dict[str(self.sl)] = (txt_encoded, txt_mask)
        if self.opts.model['include_img']:
            ctx_dict['image'] = (feats, feats_mask)

        if (self.opts.model['include_img2txt'] or
                self.opts.model['include_txt2img']):
            short_img, short_img_mask = self.img_conv(feats, feats_mask)
            short_txt, short_txt_mask = self.txt_conv(txt_encoded, txt_mask)

            img_to_txt, txt_to_img = self.coattention(
                short_txt, short_img,
                ctx_1_mask=short_txt_mask, ctx_2_mask=short_img_mask)

            if self.opts.model['include_img2txt']:
                ctx_dict['img2txt'] = (img_to_txt, short_txt_mask)
            if self.opts.model['include_txt2img']:
                ctx_dict['txt2img'] = (txt_to_img, short_img_mask)

        return ctx_dict

    def forward(self, batch, **kwargs):
        result = super().forward(batch)

        if self.training and self.opts.model['alpha_c'] > 0:
            alpha_loss = (1 - torch.cat(self.dec.alphas).sum(0)).pow(2).sum(0)
            self.aux_loss['alpha_reg'] = alpha_loss.mean().mul(
                self.opts.model['alpha_c'])

        return result
