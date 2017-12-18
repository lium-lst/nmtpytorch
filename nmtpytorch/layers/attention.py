# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Attention layer for seq2seq NMT."""
    def __init__(self, ctx_dim, hid_dim, activ='tanh', att_type='mlp'):
        super(Attention, self).__init__()

        # Get activation function
        self.activ = getattr(F, activ)

        # 'dot' or 'mlp'
        self.att_type = att_type
        assert self.att_type in ('mlp', 'dot'), \
            'Unknown attention type {}'.format(att_type)

        # Annotation dimensionality
        self.ctx_dim = ctx_dim

        # Hidden state of the RNN (or another arbitrary entity)
        self.hid_dim = hid_dim

        # MLP attention, i.e. Bahdanau et al.
        if self.att_type == 'mlp':
            self.mlp = nn.Linear(self.ctx_dim, 1, bias=False)
            self.forward = self.forward_mlp
        elif self.att_type == 'dot':
            self.forward = self.forward_dot

        # Adaptor from RNN's hidden dim to context dim
        self.hid2ctx = nn.Linear(self.hid_dim, self.ctx_dim, bias=False)

        # Additional context projection within same dimensionality
        self.ctx2ctx = nn.Linear(self.ctx_dim, self.ctx_dim, bias=False)

        # Final transformation of attended context to hidden dim
        self.ctx2hid = nn.Linear(self.ctx_dim, self.hid_dim, bias=False)

    def forward_mlp(self, hid, ctx, ctx_mask):
        r"""Computes Bahdanau-style MLP attention probabilities between
        decoder's hidden state and source annotations.

        Arguments:
            hid(Variable): A set of decoder hidden states of shape `T*B*H`
                where `T` == 1, `B` is batch dim and `H` is hidden state dim.
            ctx(Variable): A set of annotations of shape `S*B*C` where `S`
                is the source timestep dim, `B` is batch dim and `C`
                is annotation dim.
            ctx_mask(FloatTensor): A binary mask of shape `S*B` with zeroes
                in the padded timesteps.

        Returns:
            scores(Variable): A variable of shape `S*B` containing normalized
                attention scores for each position and sample.
            z_t(Variable): A variable of shape `B*H` containing the final
                attended context vector for this target decoding timestep.

        Notes:
            This will only work when `T==1` for now.
        """
        # S*B*C and T*B*C
        ctx_ = self.ctx2ctx(ctx)
        hid_ = self.hid2ctx(hid)

        # scores -> S*B
        scores = self.mlp(self.activ(ctx_ + hid_)).squeeze(-1)

        # Normalize attention scores correctly -> S*B
        alpha = (scores - scores.max(0)[0]).exp().mul(ctx_mask)
        alpha = alpha / alpha.sum(0)

        # Transform final context vector to H for further decoders
        z_t = self.ctx2hid((alpha.unsqueeze(-1) * ctx).sum(0))
        return alpha, z_t

    def forward_dot(self, hid, ctx, ctx_mask):
        r"""Computes Luong-style dot attention probabilities between
        decoder's hidden state and source annotations.

        Arguments:
            hid(Variable): A set of decoder hidden states of shape `T*B*H`
                where `T` == 1, `B` is batch dim and `H` is hidden state dim.
            ctx(Variable): A set of annotations of shape `S*B*C` where `S`
                is the source timestep dim, `B` is batch dim and `C`
                is annotation dim.
            ctx_mask(FloatTensor): A binary mask of shape `S*B` with zeroes
                in the padded timesteps.

        Returns:
            scores(Variable): A variable of shape `S*B` containing normalized
                attention scores for each position and sample.
            z_t(Variable): A variable of shape `B*H` containing the final
                attended context vector for this target decoding timestep.
        """
        # Apply transformations first to make last dims both C and then
        # shuffle dims to prepare for batch mat-mult
        ctx_ = self.ctx2ctx(ctx).permute(1, 2, 0)   # S*B*C -> S*B*C -> B*C*S
        hid_ = self.hid2ctx(hid).permute(1, 0, 2)   # T*B*H -> T*B*C -> B*T*C

        # 'dot' scores of B*T*S
        scores = F.softmax(torch.bmm(hid_, ctx_), dim=-1)

        # Transform back to hidden_dim for further decoders
        # B*T*S x B*S*C -> B*T*C -> B*T*H
        z_t = self.ctx2hid(torch.bmm(scores, ctx.transpose(0, 1)))

        return scores.transpose(0, 1), z_t.transpose(0, 1)
