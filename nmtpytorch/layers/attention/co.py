# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn

from ...utils.nn import get_activation_fn

# Code contributed by @jlibovicky


class CoAttention(nn.Module):
    """Co-attention between two sequences.

    Uses one hidden layer to compute an affinity matrix between two sequences.
    This can be then normalized in two direction which gives us 1->2 and 2->1
    attentions.

    The co-attention is computed using a single feed-forward layer as in
    Bahdanau's attention.
    """
    def __init__(self, ctx_1_dim, ctx_2_dim, bottleneck,
                 att_activ='tanh', mlp_bias=False):
        super().__init__()

        self.mlp_hid = nn.Conv2d(ctx_1_dim + ctx_2_dim, bottleneck, 1)
        self.mlp_out = nn.Conv2d(bottleneck, 1, 1, bias=mlp_bias)
        self.activ = get_activation_fn(att_activ)

        self.project_1_to_2 = nn.Linear(ctx_1_dim + ctx_2_dim, bottleneck)
        self.project_2_to_1 = nn.Linear(ctx_1_dim + ctx_2_dim, bottleneck)

    def forward(self, ctx_1, ctx_2, ctx_1_mask=None, ctx_2_mask=None):
        if ctx_2_mask is not None:
            ctx_2_neg_mask = (1. - ctx_2_mask.transpose(0, 1).unsqueeze(1)) * -1e12

        ctx_1_len = ctx_1.size(0)
        ctx_2_len = ctx_2.size(0)
        b_ctx_1 = ctx_1.permute(1, 2, 0).unsqueeze(3).repeat(1, 1, 1, ctx_2_len)
        b_ctx_2 = ctx_2.permute(1, 2, 0).unsqueeze(2).repeat(1, 1, ctx_1_len, 1)

        catted = torch.cat([b_ctx_1, b_ctx_2], dim=1)
        hidden = self.activ(self.mlp_hid(catted))
        affinity_matrix = self.mlp_out(hidden).squeeze(1)
        if ctx_1_mask is not None:
            ctx_1_neg_mask = (1. - ctx_1_mask.transpose(0, 1).unsqueeze(2)) * -1e12
            affinity_matrix += ctx_1_neg_mask

        if ctx_2_mask is not None:
            ctx_2_neg_mask = (1. - ctx_2_mask.transpose(0, 1).unsqueeze(1)) * -1e12
            affinity_matrix += ctx_2_neg_mask

        dist_1_to_2 = F.softmax(affinity_matrix, dim=2)
        context_1_to_2 = ctx_1.permute(1, 2, 0).matmul(dist_1_to_2).permute(2, 0, 1)
        seq_1_to_2 = self.activ(
            self.project_1_to_2(torch.cat([ctx_2, context_1_to_2], dim=-1)))

        dist_2_to_1 = F.softmax(affinity_matrix, dim=1).transpose(1, 2)
        context_2_to_1 = ctx_2.permute(1, 2, 0).matmul(dist_2_to_1).permute(2, 0, 1)
        seq_2_to_1 = self.activ(
            self.project_2_to_1(torch.cat([ctx_1, context_2_to_1], dim=-1)))

        return seq_2_to_1, seq_1_to_2
