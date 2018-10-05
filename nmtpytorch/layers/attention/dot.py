# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn

from ...utils.nn import get_activation_fn


class DotAttention(nn.Module):
    """Attention layer with dot product."""
    def __init__(self, ctx_dim, hid_dim, att_bottleneck='ctx',
                 transform_ctx=True, att_activ='tanh', temp=1., ctx2hid=True,
                 mlp_bias=None):
        # NOTE:
        # mlp_bias here to not break models that pass mlp_bias to all types
        # of attentions
        super().__init__()

        self.ctx_dim = ctx_dim
        self.hid_dim = hid_dim
        self._ctx2hid = ctx2hid
        self.temperature = temp
        self.activ = get_activation_fn(att_activ)

        # The common dimensionality for inner formulation
        if isinstance(att_bottleneck, int):
            self.mid_dim = att_bottleneck
        else:
            self.mid_dim = getattr(self, '{}_dim'.format(att_bottleneck))

        # Adaptor from RNN's hidden dim to mid_dim
        self.hid2ctx = nn.Linear(self.hid_dim, self.mid_dim, bias=False)

        if transform_ctx or self.mid_dim != self.ctx_dim:
            # Additional context projection within same dimensionality
            self.ctx2ctx = nn.Linear(self.ctx_dim, self.mid_dim, bias=False)
        else:
            self.ctx2ctx = lambda x: x

        if self._ctx2hid:
            # ctx2hid: final transformation from ctx to hid
            self.ctx2hid = nn.Linear(self.ctx_dim, self.hid_dim, bias=False)
        else:
            self.ctx2hid = lambda x: x

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
        # SxBxC
        ctx_ = self.ctx2ctx(ctx)
        # TxBxC
        hid_ = self.hid2ctx(hid)

        # shuffle dims to prepare for batch mat-mult -> SxB
        scores = torch.bmm(hid_.permute(1, 0, 2), ctx_.permute(1, 2, 0)).div(
            self.temperature).squeeze(1).t()

        # Normalize attention scores correctly -> S*B
        if ctx_mask is not None:
            # Mask out padded positions with -inf so that they get 0 attention
            scores.masked_fill_((1 - ctx_mask).byte(), -1e8)

        alpha = F.softmax(scores, dim=0)

        # Transform final context vector to H for further decoders
        return alpha, self.ctx2hid((alpha.unsqueeze(-1) * ctx).sum(0))
