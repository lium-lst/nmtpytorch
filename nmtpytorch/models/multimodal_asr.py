# -*- coding: utf-8 -*-
import logging

from ..layers import MultimodalBiLSTMp, ConditionalDecoder
from . import ASR

logger = logging.getLogger('nmtpytorch')


class MultimodalASR(ASR):
    """Multimodal ASR with global features + encoder/decoder initialization."""
    def set_defaults(self):
        self.defaults = {
            'feat_dim': 43,                 # Speech features dimensionality
            'emb_dim': 300,                 # Decoder embedding dim
            'enc_dim': 320,                 # Encoder hidden size
            'enc_layers': '1_1_2_2_1_1',    # layer configuration
            'dec_dim': 320,                 # Decoder hidden size
            'proj_dim': 300,                # Intra-LSTM projection layer
            'proj_activ': 'tanh',           # Intra-LSTM projection activation
            'dec_type': 'gru',              # Decoder type (gru|lstm)
            'att_type': 'mlp',              # Attention type (mlp|dot)
            'att_temp': 1.,                 # Attention temperature
            'att_activ': 'tanh',            # Attention non-linearity (all torch nonlins)
            'att_mlp_bias': False,          # Enables bias in attention mechanism
            'att_bottleneck': 'hid',        # Bottleneck dimensionality (ctx|hid)
            'att_transform_ctx': True,      # Transform annotations before attention
            'dropout': 0,                   # Generic dropout overall the architecture
            'tied_dec_embs': False,         # Share decoder embeddings
            'max_len': None,                # Reject samples if len('bucket_by') > max_len
            'bucket_by': None,              # A key like 'en' to define w.r.t
                                            # which dataset batches will be sorted
            'bucket_order': None,           # Can be 'ascending' or 'descending'
                                            # for curriculum learning
            'sampler_type': 'bucket',       # bucket or approximate. latter is not good
                                            # since the ASR encoder does not handle paddings
            'sched_sampling': 0,            # Scheduled sampling ratio
            'direction': None,              # Network directionality, i.e. en->de
            'lstm_forget_bias': False,      # Initialize forget gate bias to 1 for LSTM
            'lstm_bias_zero': False,        # Use zero biases for LSTM
            'dec_init': 'mean_ctx',         # How to initialize decoder
                                            # (zero/mean_ctx/feats)
            'dec_init_size': None,          # feature vector dimensionality for
                                            # dec_init == 'feats'
            'dec_init_activ': 'tanh',       # Decoder initialization activation func
            'aux_dim': 2048,                # Feature dimension for multimodal encoder
            'feat_activ': None,             # Feature non-linearity for multimodal encoder
            'feat_fusion': 'init',          # Integration type for multimodal encoder
            'tied_init': False,             # Tie FFs for enc and dec init
            'bos_type': 'emb',              # 'emb': classical learned <bos>
                                            # 'feats': use visual feats as <bos>
            'bos_activ': None,              # activation function for 'feats'
            'bos_dim': None,                # input feats dim for bos 'feats'
            'bos_bias': False,              # bias for bos 'feats'
        }

    def __init__(self, opts):
        super().__init__(opts)

    def setup(self, is_train=True):
        self.speech_enc = MultimodalBiLSTMp(
            input_size=self.opts.model['feat_dim'],
            hidden_size=self.opts.model['enc_dim'],
            proj_size=self.opts.model['proj_dim'],
            proj_activ=self.opts.model['proj_activ'],
            dropout=self.opts.model['dropout'],
            layers=self.opts.model['enc_layers'],
            feat_size=self.opts.model['aux_dim'],
            feat_activ=self.opts.model['feat_activ'],
            feat_fusion=self.opts.model['feat_fusion'])

        ################
        # Create Decoder
        ################
        self.dec = ConditionalDecoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name=str(self.src),
            tied_emb=self.opts.model['tied_dec_embs'],
            dec_init=self.opts.model['dec_init'],
            dec_init_size=self.opts.model['dec_init_size'],
            dec_init_activ=self.opts.model['dec_init_activ'],
            att_type=self.opts.model['att_type'],
            att_temp=self.opts.model['att_temp'],
            att_activ=self.opts.model['att_activ'],
            transform_ctx=self.opts.model['att_transform_ctx'],
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout'],
            sched_sample=self.opts.model['sched_sampling'],
            bos_type=self.opts.model['bos_type'],
            bos_dim=self.opts.model['bos_dim'],
            bos_activ=self.opts.model['bos_activ'],
            bos_bias=self.opts.model['bos_bias'])

        if self.opts.model['dec_init'] == 'feats' and self.opts.model['tied_init']:
            # Use the same FF for enc and dec init layers
            self.speech_enc.ff_init_h0.weight = self.dec.ff_dec_init.weight
            # Tie the <bos> 'feats' layer as well
            if self.opts.model['bos_type'] == 'feats':
                self.dec.ff_bos.weight = self.dec.ff_dec_init.weight

    def encode(self, batch, **kwargs):
        d = {str(self.src): self.speech_enc(batch[self.src], aux=batch['feats'])}

        if 'feats' in batch:
            d['feats'] = (batch['feats'], None)
        return d
