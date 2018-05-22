# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attentionv2(nn.Module):
    """Attention mechanism for sequence-to-sequence models. The MLP method
    is equivalent to the classical content-based attention.

    Arguments:
        ctx_dim(int): The dimensionality of source hidden states over which
            the attention will be applied.
        hid_dim(int): The dimensionality of decoder's hidden states that
            will be used as attention queries.
        mid_dim(int|str): The common dimensionality that
            source hidden states and decoder hidden states should be projected.
            This may also be one of 'ctx' or 'hid' to select ``ctx_dim`` or
            ``hid_dim``.
        method(str, optional): Use ``mlp`` (default) or ``dot`` attention.
        mlp_activ(str, optional): If ``method == 'mlp'``, defines which
            non-linearity to use before the transformation. Default is ``tanh``.
        mlp_bias(bool, optional): If ``method == 'mlp'``, defines whether
            the layer should have a bias or no. Default is ``False``.
        temp(float, optional): Sharpen the softmax distribution to narrow
            its focus. Default is ``1``, i.e. not enabled.
        final_ctx_transform(bool, optional): If ``False``, the final weighted-sum
            context vector of ``ctx_dim`` will not be linearly transformed
            to ``hid_dim``.
    """

    def __init__(self, ctx_dim, hid_dim, mid_dim,
                 method='mlp', mlp_activ='tanh', mlp_bias=False,
                 temp=1., final_ctx_transform=True):
        super().__init__()

        self.ctx_dim = ctx_dim
        self.hid_dim = hid_dim
        self.method = method
        self.mlp_activ = mlp_activ
        self.mlp_bias = mlp_bias
        self.temp = temp
        self.final_ctx_transform = final_ctx_transform
        self.activ = getattr(F, self.mlp_activ)

        # The common dimensionality for inner formulation
        if isinstance(mid_dim, int):
            self.mid_dim = mid_dim
        else:
            self.mid_dim = getattr(self, '{}_dim'.format(mid_dim))

        if self.method == 'mlp':
            self.ff = nn.Linear(self.mid_dim, 1, bias=False)
            self._scores = self._mlp_scores
        elif self.method == 'dot':
            self._scores = self._dot_scores
        else:
            raise RuntimeError('Unknown attention: {}'.format(self.method))

        # Adaptor from decoder's hidden dim to mid_dim
        self.hid2mid = nn.Linear(self.hid_dim, self.mid_dim, bias=False)

        # Adaptor from encoder's hidden dim to mid_dim
        self.ctx2mid = nn.Linear(self.ctx_dim, self.mid_dim, bias=False)

        # (Optional) adapter from weighted-sum ctx vector to hid_dim
        if self.final_ctx_transform:
            self.att2hid = nn.Linear(self.ctx_dim, self.hid_dim, bias=False)
        else:
            self.att2hid = lambda x: x

    def forward(self, hid, ctx, ctx_mask=None):
        r"""Computes attention probabilities and final context using
        decoder's hidden state and source annotations.

        Arguments:
            hid(Variable): A set of decoder hidden states of shape `T*B*H`
                where `T` == target_seq_len, `B` is batch and `H` is hid_dim.
            ctx(Variable): A set of encodings of shape `S*B*C` where `S`
                is the source seq_len, `B` is batch dim and `C` is ctx_dim.
            ctx_mask(FloatTensor): A binary mask of shape `S*B` with zeroes
                in the empty/padded positions. This will be ``None`` if the
                source encoder processes batches with equal-length samples.

        Returns:
            scores(Variable): A variable of shape `T*S*B` containing normalized
                attention scores for each position and sample.
            z_t(Variable): A variable of shape `T*B*H` containing the final
                attended context vector for target decoder hidden state(s).

        Notes:
            This should work for single timesteps and many timesteps if you
                ensure that the ``T == 1`` in the former case.
        """
        scores = self._scores(
            self.ctx2mid(ctx),    # S*B*C -> S*B*M
            self.hid2mid(hid),    # T*B*H -> T*B*M
        ).div(self.temp)          # = S*B*T

        if ctx_mask is not None:
            # Mask out padded positions with -inf so that they get 0 attention
            scores.masked_fill_((1 - ctx_mask).unsqueeze(-1).byte(), -1e8)

        # Normalize attention scores (S*B*T) correctly (softmax dim=S)
        # S*B*T -> B*T*S
        alpha = F.softmax(scores, dim=0).permute(1, 2, 0)

        # B*T*S x B*S*C -> B*T*C -> T*B*C
        attended_ctx = torch.bmm(alpha, ctx.transpose(0, 1)).transpose(0, 1)

        return alpha.permute(1, 2, 0), self.att2hid(attended_ctx)

    def _dot_scores(self, ctx_, hid_):
        # shuffle dims to prepare for batch mat-mult
        return torch.bmm(
            ctx_.transpose(0, 1),    # B*S*M
            hid_.permute(1, 2, 0),   # B*M*T
        ).transpose(0, 1)            # B*S*T -> S*B*T

    def _mlp_scores(self, ctx_, hid_):
        # Expand context to target steps and sum with decoder states
        # S*T*B*M = S*T*B*M + S*1*B*M
        summed = hid_.expand(ctx_.shape[0], -1, -1, -1) + ctx_.unsqueeze(1)

        # What follows is: S*B*T
        return self.ff(self.activ(summed)).squeeze(-1).transpose(1, 2)
