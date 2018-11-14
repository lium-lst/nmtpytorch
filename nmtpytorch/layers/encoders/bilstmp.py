# -*- coding: utf-8 -*-
import logging
from torch import nn
from torch.nn import functional as F

from ..ff import FF

logger = logging.getLogger('nmtpytorch')


class BiLSTMp(nn.Module):
    """A bidirectional LSTM encoder for speech features. A batch should
    only contain samples that have the same sequence length.

    Arguments:
        input_size (int): Input feature dimensionality.
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
    def __init__(self, input_size, hidden_size, proj_size, layers,
                 proj_activ='tanh', dropout=0):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.proj_activ = proj_activ
        self.layers = [int(i) for i in layers.split('_')]
        self.dropout = dropout
        self.n_layers = len(self.layers)

        # Doubles its size because of concatenation of forw-backw encs
        self.ctx_size = self.hidden_size * 2

        # Fill 0-vector as <eos> to the end of the frames
        self.pad_tuple = (0, 0, 0, 0, 0, 1)

        # Projections and LSTMs
        self.ffs = nn.ModuleList()
        self.lstms = nn.ModuleList()

        if self.dropout > 0:
            self.do = nn.Dropout(self.dropout)

        for i, ss_factor in enumerate(self.layers):
            # Add LSTMs
            self.lstms.append(nn.LSTM(
                self.input_size if i == 0 else self.hidden_size,
                self.hidden_size, bidirectional=True))
            # Add non-linear bottlenecks
            self.ffs.append(FF(
                self.ctx_size, self.proj_size, activ=self.proj_activ))

    def forward(self, x, **kwargs):
        # Generate a mask to detect padded sequences
        mask = x.ne(0).float().sum(2).ne(0).float()

        if mask.eq(0).nonzero().numel() > 0:
            logger.info("WARNING: Non-homogeneous batch in BiLSTMp.")

        # Pad with <eos> zero
        hs = F.pad(x, self.pad_tuple)

        for (ss_factor, f_lstm, f_ff) in zip(self.layers, self.lstms, self.ffs):
            if ss_factor > 1:
                # Skip states
                hs = f_ff(f_lstm(hs[::ss_factor])[0])
            else:
                hs = f_ff(f_lstm(hs)[0])

        if self.dropout > 0:
            hs = self.do(hs)

        # No mask is returned as batch should contain same-length sequences
        return hs, None
