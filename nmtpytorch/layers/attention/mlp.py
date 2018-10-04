# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn

from .dot import DotAttention


class MLPAttention(DotAttention):
    """Attention layer with feed-forward layer."""
    def __init__(self, ctx_dim, hid_dim, att_bottleneck='ctx',
                 transform_ctx=True, att_activ='tanh',
                 mlp_bias=False, temp=1., ctx2hid=True):
        super().__init__(ctx_dim, hid_dim, att_bottleneck, transform_ctx,
                         att_activ, temp, ctx2hid)

        if mlp_bias:
            self.bias = nn.Parameter(torch.Tensor(self.mid_dim))
            self.bias.data.zero_()
        else:
            self.register_parameter('bias', None)

        self.mlp = nn.Linear(self.mid_dim, 1, bias=False)

    def forward(self, hid, ctx, ctx_mask=None):
        r"""Computes attention probabilities and final context using
        decoder's hidden state and source annotations.

        Arguments:
            hid(Tensor): A set of decoder hidden states of shape `T*B*H`
                where `T` == 1, `B` is batch dim and `H` is hidden state dim.
            ctx(Tensor): A set of annotations of shape `S*B*C` where `S`
                is the source timestep dim, `B` is batch dim and `C`
                is annotation dim.
            ctx_mask(FloatTensor): A binary mask of shape `S*B` with zeroes
                in the padded positions.

        Returns:
            scores(Tensor): A tensor of shape `S*B` containing normalized
                attention scores for each position and sample.
            z_t(Tensor): A tensor of shape `B*H` containing the final
                attended context vector for this target decoding timestep.

        Notes:
            This will only work when `T==1` for now.
        """
        # inner_sum -> SxBxC + TxBxC
        inner_sum = self.ctx2ctx(ctx) + self.hid2ctx(hid)

        if self.bias is not None:
            inner_sum.add_(self.bias)

        # Compute scores- > SxB
        scores = self.mlp(
            self.activ(inner_sum)).div(self.temperature).squeeze(-1)

        # Normalize attention scores correctly -> S*B
        if ctx_mask is not None:
            # Mask out padded positions with -inf so that they get 0 attention
            scores.masked_fill_((1 - ctx_mask).byte(), -1e8)

        alpha = F.softmax(scores, dim=0)

        # Transform final context vector to H for further decoders
        return alpha, self.ctx2hid((alpha.unsqueeze(-1) * ctx).sum(0))
