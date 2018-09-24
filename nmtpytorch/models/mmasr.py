# -*- coding: utf-8 -*-
import logging

from ..layers import BiLSTMp, ConditionalMMDecoder, FF
from .asr import ASR

logger = logging.getLogger('nmtpytorch')


# A multimodal/attentive ASR with video-level features


class MultimodalASR(ASR):
    supports_beam_search = True

    def set_defaults(self):
        # Set parent defaults
        super().set_defaults()
        self.defaults.update({
            'fusion_type': 'concat',  # Multimodal context fusion (sum|mul|concat)
            'visual_dim': 2048,       # Visual feature dimensionality
            'visual_seq': True,       # True if you have a set of visual features
            'proj_dim': None,         # Project visual features to some speech hidden dim
        })

    def __init__(self, opts):
        super().__init__(opts)

    def setup(self, is_train=True):
        if self.opts.model['proj_dim'] is not None:
            self.proj_layer = FF(self.opts.model['visual_dim'],
                                 self.opts.model['proj_dim'])
            self.vis_ctx_dim = self.opts.model['proj_dim']
            self.ctx_sizes['vfeats'] = self.opts.model['proj_dim']
        else:
            self.ctx_sizes['vfeats'] = self.opts.model['visual_dim']

        self.speech_enc = BiLSTMp(
            input_size=self.opts.model['feat_dim'],
            hidden_size=self.opts.model['enc_dim'],
            proj_size=self.opts.model['proj_dim'],
            proj_activ=self.opts.model['proj_activ'],
            dropout=self.opts.model['dropout'],
            layers=self.opts.model['enc_layers'])

        ################
        # Create Decoder
        ################
        self.dec = ConditionalMMDecoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name=str(self.src),
            aux_ctx_name='vfeats',
            fusion_type=self.opts.model['fusion_type'],
            tied_emb=self.opts.model['tied_dec_embs'],
            dec_init=self.opts.model['dec_init'],
            att_type=self.opts.model['att_type'],
            att_activ=self.opts.model['att_activ'],
            transform_ctx=self.opts.model['att_transform_ctx'],
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout'])

    def encode(self, batch, **kwargs):
        # Let's start with a None mask by assuming that
        # we have a fixed-length feature collection
        feats_mask = None

        # Be it Numpy or NumpySequence, they return
        # (n_samples, feat_dim, t) by default
        # Convert it to (t, n_samples, feat_dim)
        feats = batch['vfeats'].view(
            (*batch['vfeats'].shape[:2], -1)).permute(2, 0, 1)

        if self.opts.model['visual_seq']:
            # Let's create mask in this case
            feats_mask = feats.ne(0).float().sum(2).ne(0).float()

        if self.opts.model['proj_dim']:
            feats = self.proj_layer(feats)

        return {
            'vfeats': (feats, feats_mask),
            str(self.src): self.speech_enc(batch[self.src]),
        }
