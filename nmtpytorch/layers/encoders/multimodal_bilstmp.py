# -*- coding: utf-8 -*-
import logging

import torch
from torch.nn import functional as F

from ..ff import FF

from . import BiLSTMp

logger = logging.getLogger('nmtpytorch')


class MultimodalBiLSTMp(BiLSTMp):
    """A bidirectional multimodal LSTM encoder for speech features.

    Arguments:
        feat_size (int): Auxiliary feature dimensionality.
        feat_fusion(str): Type of feature fusion: 'early_concat', 'early_sum',
            'late_concat', 'late_sum', 'init'.
        feat_activ(str): Type of non-linearity if any for feature projection
            layer.
        input_size (int): Input speech feature dimensionality.
        hidden_size (int): LSTM hidden state dimensionality.
        proj_size (int): Projection layer size.
        proj_activ (str, optional): Non-linearity to apply to intermediate projection
            layers. (Default: 'tanh')
        layers (str): A '_' separated list of integers that defines the subsampling
            factor for each LSTM.
        dropout (float, optional): Use dropout (Default: 0.)
    Input:
        x (Tensor): A tensor of shape (n_timesteps, n_samples, n_feats)
            that includes acoustic features of dimension ``n_feats`` per
            each timestep (in the first dimension).

    Output:
        hs (Tensor): A tensor of shape (n_timesteps, n_samples, hidden * 2)
            that contains encoder hidden states for all timesteps.
        mask (Tensor): `None` since this layer expects all equal frame inputs.
    """

    def __init__(self, feat_size, feat_fusion, feat_activ=None, **kwargs):
        # Call BiLSTMp.__init__ first
        super().__init__(**kwargs)

        self.feat_size = feat_size
        self.feat_fusion = feat_fusion
        self.feat_activ = feat_activ

        # early_concat: x = layer([x; aux])
        #  layer: feat_size + input_size -> input_size
        if self.feat_fusion == 'early_concat':
            self.feat_layer = FF(
                self.feat_size + self.input_size, self.input_size, activ=self.feat_activ)
        # early_sum: x = x + layer(aux)
        #  layer: feat_size -> input_size
        elif self.feat_fusion == 'early_sum':
            self.feat_layer = FF(self.feat_size, self.input_size, activ=self.feat_activ)
        # late_concat: hs = layer([hs; aux])
        #  layer: proj_size + feat_size -> proj_size
        elif self.feat_fusion == 'late_concat':
            self.feat_layer = FF(
                self.feat_size + self.proj_size, self.proj_size, activ=self.feat_activ)
        # late_sum: hs = hs + layer(aux)
        #  layer: feat_size -> proj_size
        elif self.feat_fusion == 'late_sum':
            self.feat_layer = FF(self.feat_size, self.proj_size, activ=self.feat_activ)
        # init: Initialize all LSTMs
        elif self.feat_fusion == 'init':
            # Use single h_0/c_0 for all stacked layers and directions for a
            # consistent information source.
            self.ff_init_c0 = FF(self.feat_size, self.hidden_size, activ=self.feat_activ)
            self.ff_init_h0 = FF(self.feat_size, self.hidden_size, activ=self.feat_activ)

    def forward(self, x, **kwargs):
        # Generate a mask to detect padded sequences
        mask = x.ne(0).float().sum(2).ne(0).float()

        if mask.eq(0).nonzero().numel() > 0:
            logger.info("WARNING: Non-homogeneous batch in BiLSTMp.")

        # Get auxiliary input
        aux_x = kwargs['aux']

        ##############
        # Encoder init
        ##############
        if self.feat_fusion == 'init':
            # Tile to 2xBxH for bidirectionality
            c_0_ = self.ff_init_c0(aux_x).repeat(2, 1, 1)
            h_0_ = self.ff_init_h0(aux_x).repeat(2, 1, 1)

            # Should be a tuple of (h, c) for each layer
            h_0s = [(h_0_, c_0_) for _ in range(self.n_layers)]
        else:
            # Dummy setup so that the below method calls are good
            h_0s = [None for _ in range(self.n_layers)]
            if self.feat_fusion == 'early_concat':
                x = self.feat_layer(
                    torch.cat([x, aux_x.repeat(x.shape[0], 1, 1)], dim=-1))
            elif self.feat_fusion == 'early_sum':
                x.add_(self.feat_layer(aux_x).unsqueeze(0))

        # Pad with <eos> zero
        hs = F.pad(x, self.pad_tuple)

        ###################
        # LSTM + Proj block
        ###################
        for (ss_factor, f_lstm, f_ff, h_0) in zip(self.layers, self.lstms, self.ffs, h_0s):
            if ss_factor > 1:
                # Skip states
                hs = f_ff(f_lstm(hs[::ss_factor], hx=h_0)[0])
            else:
                hs = f_ff(f_lstm(hs, hx=h_0)[0])

        #############
        # Late Fusion
        #############
        if self.feat_fusion == 'late_concat':
            hs = self.feat_layer(
                torch.cat([hs, aux_x.repeat(hs.shape[0], 1, 1)], dim=-1))
        elif self.feat_fusion == 'late_sum':
            hs = hs + self.feat_layer(aux_x).unsqueeze(0)

        if self.dropout > 0:
            hs = self.do(hs)

        # No mask is returned as batch should contain same-length sequences
        return hs, None
