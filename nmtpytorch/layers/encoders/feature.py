# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from .. import FF

# Initially implemented for frame-level video feature encoder, this is
# actually a generic sequential encoder without explicit embeddings.
# Code contributed by @elliottd


class FeatureEncoder(nn.Module):
    """A recurrent feature encoder without explicit embeddings.

    Arguments:
        input_size (int): Input feature dimensionality.
        proj_size (int): Intermediate projection dimensionality before RNN.
        hidden_size (int): RNN hidden state dimensionality.
        rnn_type (str): RNN Type, i.e. GRU or LSTM.
        proj_activ(str, optional): Non-linearity type for intermediate
            projection (Default: 'tanh').
        num_layers (int, optional): Number of stacked RNNs (Default: 1).
        bidirectional (bool, optional): If `False`, the RNN is unidirectional.
        dropout_rnn (float, optional): Inter-layer dropout rate only
            applicable if `num_layers > 1`. (Default: 0.)
        dropout_emb(float, optional): Dropout rate for embeddings (Default: 0.)
        dropout_ctx(float, optional): Dropout rate for the hidden states (Default: 0.)

    Input:
        x (Tensor): A tensor of shape (n_timesteps, n_samples)
            including the integer token indices for the given batch.
    Output:
        hs (Tensor): A tensor of shape (n_timesteps, n_samples, hidden)
            that contains encoder hidden states for all timesteps. If
            bidirectional, `hs` is doubled in size in the last dimension
            to contain both directional states.
        mask (Tensor): A binary mask of shape (n_timesteps, n_samples)
            that may further be used in attention and/or decoder.
    """
    def __init__(self, input_size, proj_size, hidden_size, rnn_type,
                 proj_activ='tanh', num_layers=1, bidirectional=True,
                 dropout_rnn=0, dropout_emb=0, dropout_ctx=0):
        super().__init__()

        self.rnn_type = rnn_type.upper()
        self.input_size = input_size
        self.proj_size = proj_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # For dropout btw layers, only effective if num_layers > 1
        self.dropout_rnn = dropout_rnn

        # Our other custom dropouts after embeddings and annotations
        self.dropout_emb = dropout_emb
        self.dropout_ctx = dropout_ctx

        self.ctx_size = self.hidden_size
        # Doubles its size because of concatenation
        if self.bidirectional:
            self.ctx_size *= 2

        if self.dropout_emb > 0:
            self.do_emb = nn.Dropout(self.dropout_emb)
        if self.dropout_ctx > 0:
            self.do_ctx = nn.Dropout(self.dropout_ctx)

        # Create a video frame embedding layer that maps 2048 -> proj_size
        # This is an atypical embedding: add a bias + non-linear activation
        self.emb = FF(
            self.input_size, self.proj_size, bias=True, activ=proj_activ)

        # Create fused/cudnn encoder according to the requested type
        RNN = getattr(nn, self.rnn_type)
        self.enc = RNN(self.proj_size, self.hidden_size,
                       self.num_layers, bias=True, batch_first=False,
                       dropout=self.dropout_rnn,
                       bidirectional=self.bidirectional)

    def forward(self, x, **kwargs):
        # Embed the video feature vectors using a Linear layer
        proj = self.emb(x.float())

        # Get the mask
        mask = proj.ne(0).float().sum(2).ne(0).float()

        if mask.eq(0).nonzero().numel() > 0:
            # padded with zeros
            slens, sidxs = mask.sum(0).sort(descending=True)
            old_idxs = torch.sort(sidxs)[1]
            # reorder
            proj = proj[:, sidxs]
            if self.dropout_emb > 0:
                proj = self.do_emb(proj)

            # Pack and encode
            packed_input = pack_padded_sequence(proj, slens.long().data.tolist())
            packed_hs, _ = self.enc(packed_input)
            # Get hidden states and revert the order
            hs = pad_packed_sequence(packed_hs)[0][:, old_idxs]
        else:
            # all equal
            mask = None
            if self.dropout_emb > 0:
                proj = self.do_emb(proj)

            # Encode
            hs, _ = self.enc(proj)

        if self.dropout_ctx > 0:
            hs = self.do_ctx(hs)

        return hs, mask
