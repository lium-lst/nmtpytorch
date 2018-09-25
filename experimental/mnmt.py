# -*- coding: utf-8 -*-
import logging


from ..layers import MNMTEncoder, ConditionalDecoder, FF
from .nmt import NMT

logger = logging.getLogger('nmtpytorch')


class MNMT(NMT):
    supports_beam_search = True

    def set_defaults(self):
        # Set parent defaults
        super().set_defaults()
        self.defaults.update({
            'feat_dim': None,         # feat_dim should be given from the config
            'feat_fusion': 'concat',  # concatenate with source embeddings or
                                      # 'prepend' to source embedding sequence
        })

    def __init__(self, opts):
        super().__init__(opts)

        # Safety check
        # This should either be 1000 for softmax features or
        # 2048 for pool5 features
        assert self.opts.model['feat_dim'] is not None, \
            "feat_dim should be given in the configuration file."

    def setup(self, is_train=True):
        # This encoder takes an additional parameter 'feat_fusion'
        # that is given in the configuration file.
        self.enc = MNMTEncoder(
            emb_dim=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['enc_dim'],
            n_vocab=self.n_src_vocab,
            rnn_type=self.opts.model['enc_type'],
            dropout_emb=self.opts.model['dropout_emb'],
            dropout_ctx=self.opts.model['dropout_ctx'],
            dropout_rnn=self.opts.model['dropout_enc'],
            num_layers=self.opts.model['n_encoders'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            feat_fusion=self.opts.model['feat_fusion'],
        )

        # Decoder is the same as NMT decoder
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
            emb_gradscale=self.opts.model['emb_gradscale'])

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            self.enc.emb.weight = self.dec.emb.weight

        # feed-forward (FF) layer to project feat_dim to emb_dim
        # which is what concat/prepend
        self.proj = FF(self.opts.model['feat_dim'], self.opts.model['emb_dim'])

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

        # Project visual features from the batch using the FF layer
        feats = self.proj(batch['image'])

        # Pass the projected features to mnmt encoder
        return {
            str(self.sl): self.enc(batch[self.sl], feats),
        }
