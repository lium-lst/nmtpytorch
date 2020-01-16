# -*- coding: utf-8 -*-
from ..logger import Logger

import torch

from ..datasets import MultimodalDataset
from ..layers import ConditionalMMDecoder, TextEncoder
from .nmt import NMT

log = Logger()


class AttentiveMNMTFeatures(NMT):
    """An end-to-end sequence-to-sequence NMT model with visual attention over
    pre-extracted convolutional features.
    """
    def set_defaults(self):
        # Set parent defaults
        super().set_defaults()
        self.defaults.update({
            'fusion_type': 'concat',    # Multimodal context fusion (sum|mul|concat)
            'fusion_activ': None,       # Multimodal context non-linearity
            'n_channels': 2048,         # depends on the features used
            'alpha_c': 0.0,             # doubly stoch. attention
            'mm_att_type': 'md-dd',     # multimodal attention type
                                        # md: modality dep.
                                        # mi: modality indep.
                                        # dd: decoder state dep.
                                        # di: decoder state indep.
            'out_logic': 'simple',      # simple vs deep output
            'persistent_dump': False,   # To save activations during beam-search
            'img_sequence': False,      # if true img is sequence of img features,
                                        # otherwise it's a conv map
        })

    def __init__(self, opts):
        super().__init__(opts)
        if self.opts.model['alpha_c'] > 0:
            self.aux_loss['alpha_reg'] = 0.0

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
            att_ctx2hid=self.opts.model['att_ctx2hid'],
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout_out'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            persistent_dump=self.opts.model['persistent_dump'])

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
        log.log(dataset)
        return dataset

    def encode(self, batch, **kwargs):
        # Let's start with a None mask by assuming that
        # we have a fixed-length feature collection
        feats, feats_mask = batch['image'], None

        if self.opts.model['img_sequence']:
            # Let's create mask in this case
            feats_mask = feats.ne(0).float().sum(2).ne(0).float()

        return {
            'image': (feats, feats_mask),
            str(self.sl): self.enc(batch[self.sl]),
        }

    def forward(self, batch, **kwargs):
        result = super().forward(batch)

        if self.training and self.opts.model['alpha_c'] > 0:
            alpha_loss = (
                1 - torch.cat(self.dec.history['alpha_img']).sum(0)).pow(2).sum(0)
            self.aux_loss['alpha_reg'] = alpha_loss.mean().mul(
                self.opts.model['alpha_c'])

        return result
