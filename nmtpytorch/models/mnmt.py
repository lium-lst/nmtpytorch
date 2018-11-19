# -*- coding: utf-8 -*-
import logging

from .nmt import NMT

logger = logging.getLogger('nmtpytorch')


class MultimodalNMT(NMT):
    """A encoder/decoder enriched multimodal NMT.

        Integration types (feat_fusion argument)
            'encinit':      Initialize RNNs in the encoder
            'decinit':      Initializes first decoder RNN.
            'vbos':         Replace <bos> in decoder with visual features
            'concat':       Concat the embeddings and features (doubles RNN input)
            'concat_fuse':  Concat the embeddings and features and project
            'sum':          Sum the embeddings with projected features
            'prepend':      Input sequence: [vis, embs, eos]
            'append':       Input sequence: [embs, vis, eos]
            'preappend':    Input sequence: [vis, embs, vis, eos]
            'vis_replace':  Replaces '<vis>' embedding with features
    """
    def __init__(self, opts):
        super().__init__(opts)

    def set_defaults(self):
        # Set parent defaults
        super().set_defaults()
        self.defaults.update({
            'feat_dim': 2048,       # Feature dimension for multimodal encoder
            'feat_activ': None,     # Feature non-linearity for multimodal encoder
            'feat_fusion': 'encinit',   # By default initialize only the encoder
        })

    def setup(self, is_train=True):
        """Sets up NN topology by creating the layers."""
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
            feat_size=self.opts.model['feat_dim'],
            feat_activ=self.opts.model['feat_activ'],
            feat_fusion=self.opts.model['feat_fusion'])

        self.dec = ConditionalDecoder(
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
            transform_ctx=self.opts.model['att_transform_ctx'],
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout_out'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            sched_sample=self.opts.model['sched_sampling'])

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            self.enc.emb.weight = self.dec.emb.weight

    def encode(self, batch, **kwargs):
        # BxD auxiliary features
        aux_feats = batch['feats']

        # By default, there is encoder-side integration
        d = {str(self.sl): self.enc(batch[self.sl], aux=aux_feats)}

        # It may also be decoder-side integration
        if self.opts.model['dec_init'] == 'feats':
            d['feats'] = (vis_feats, None)
        return d
