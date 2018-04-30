# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Attention layer for seq2seq NMT."""
    def __init__(self, ctx_dim, hid_dim, att_bottleneck='ctx',
                 transform_ctx=True, att_activ='tanh', att_type='mlp',
                 mlp_bias=False, temp=1., ctx2hid=True):
        super().__init__()

        self.activ = getattr(F, att_activ)
        self.ctx_dim = ctx_dim
        self.hid_dim = hid_dim
        self.att_type = att_type
        self.temp = temp
        self._ctx2hid = ctx2hid

        # The common dimensionality for inner formulation
        if isinstance(att_bottleneck, int):
            self.mid_dim = att_bottleneck
        else:
            self.mid_dim = getattr(self, '{}_dim'.format(att_bottleneck))

        if self.att_type == 'mlp':
            self.mlp = nn.Linear(self.mid_dim, 1, bias=mlp_bias)
            self._scores = self._mlp_scores
        elif self.att_type == 'dot':
            self._scores = self._dot_scores
        else:
            raise Exception('Unknown attention type {}'.format(att_type))

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
            hid(Variable): A set of decoder hidden states of shape `T*B*H`
                where `T` == 1, `B` is batch dim and `H` is hidden state dim.
            ctx(Variable): A set of annotations of shape `S*B*C` where `S`
                is the source timestep dim, `B` is batch dim and `C`
                is annotation dim.
            ctx_mask(FloatTensor): A binary mask of shape `S*B` with zeroes
                in the padded positions.

        Returns:
            scores(Variable): A variable of shape `S*B` containing normalized
                attention scores for each position and sample.
            z_t(Variable): A variable of shape `B*H` containing the final
                attended context vector for this target decoding timestep.

        Notes:
            This will only work when `T==1` for now.
        """
        # scores -> SxB
        scores = self._scores(self.ctx2ctx(ctx),    # SxBxC
                              self.hid2ctx(hid))    # TxBxC

        # Normalize attention scores correctly -> S*B
        if ctx_mask is None:
            alpha = F.softmax(scores, dim=0)
        else:
            alpha = (scores - scores.max(0)[0]).exp().mul(ctx_mask)
            alpha = alpha / alpha.sum(0)

        # Transform final context vector to H for further decoders
        return alpha, self.ctx2hid((alpha.unsqueeze(-1) * ctx).sum(0))

    def _dot_scores(self, ctx_, hid_):
        # shuffle dims to prepare for batch mat-mult
        return torch.bmm(hid_.permute(1, 0, 2), ctx_.permute(1, 2, 0)).div(
            self.temp).squeeze(1).t()

    def _mlp_scores(self, ctx_, hid_):
        return self.mlp(self.activ(ctx_ + hid_)).div(self.temp).squeeze(-1)
