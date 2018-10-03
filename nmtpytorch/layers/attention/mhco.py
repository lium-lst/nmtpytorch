# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn

# Code contributed by @jlibovicky


class MultiHeadCoAttention(nn.Module):
    """Generalization of multi-head attention for co-attention."""

    def __init__(self, ctx_1_dim, ctx_2_dim, bottleneck, head_count, dropout=0.1):
        assert bottleneck % head_count == 0
        self.dim_per_head = bottleneck // head_count
        self.model_dim = bottleneck

        super().__init__()
        self.head_count = head_count

        self.linear_keys_1 = nn.Linear(ctx_1_dim,
                                       head_count * self.dim_per_head)
        self.linear_values_1 = nn.Linear(ctx_1_dim,
                                         head_count * self.dim_per_head)
        self.linear_keys_2 = nn.Linear(ctx_2_dim,
                                       head_count * self.dim_per_head)
        self.linear_values_2 = nn.Linear(ctx_2_dim,
                                         head_count * self.dim_per_head)

        self.final_1_to_2_linear = nn.Linear(bottleneck, bottleneck)
        self.final_2_to_1_linear = nn.Linear(bottleneck, bottleneck)
        self.project_1_to_2 = nn.Linear(ctx_1_dim + ctx_2_dim, bottleneck)
        self.project_2_to_1 = nn.Linear(ctx_1_dim + ctx_2_dim, bottleneck)

    def forward(self, ctx_1, ctx_2, ctx_1_mask=None, ctx_2_mask=None):
        """Computes the context vector and the attention vectors."""

        def shape(x, length):
            """  projection """
            return x.view(
                length, batch_size, head_count, dim_per_head).permute(1, 2, 0, 3)

        def unshape(x, length):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(
                batch_size, length, head_count * dim_per_head).transpose(0, 1)

        batch_size = ctx_1.size(1)
        assert batch_size == ctx_2.size(1)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        ctx_1_len = ctx_1.size(0)
        ctx_2_len = ctx_2.size(0)

        # 1) Project key, value, and key_2.
        key_1_up = shape(self.linear_keys_1(ctx_1), ctx_1_len)
        value_1_up = shape(self.linear_values_1(ctx_1), ctx_1_len)
        key_2_up = shape(self.linear_keys_2(ctx_2), ctx_2_len)
        value_2_up = shape(self.linear_values_2(ctx_2), ctx_2_len)

        scores = torch.matmul(key_2_up, key_1_up.transpose(2, 3))

        if ctx_1_mask is not None:
            mask = ctx_1_mask.t().unsqueeze(2).unsqueeze(3).expand_as(scores)
            scores = scores.masked_fill(mask.byte(), -1e18)
        if ctx_2_mask is not None:
            mask = ctx_2_mask.t().unsqueeze(1).unsqueeze(3).expand_as(scores)
            scores = scores.masked_fill(mask.byte(), -1e18)

        # 3) Apply attention dropout and compute context vectors.
        dist_1_to_2 = F.softmax(scores, dim=2)
        context_1_to_2 = unshape(torch.matmul(dist_1_to_2, value_1_up), ctx_2_len)
        context_1_to_2 = self.final_1_to_2_linear(context_1_to_2)
        seq_1_to_2 = self.activ(
            self.project_1_to_2(torch.cat([ctx_2, context_1_to_2], dim=-1)))

        dist_2_to_1 = F.softmax(scores, dim=1)
        context_2_to_1 = unshape(
            torch.matmul(dist_2_to_1.transpose(2, 3), value_2_up), ctx_1_len)
        context_2_to_1 = self.final_2_to_1_linear(context_2_to_1)
        seq_2_to_1 = self.activ(
            self.project_2_to_1(torch.cat([ctx_1, context_2_to_1], dim=-1)))

        return seq_2_to_1, seq_1_to_2
