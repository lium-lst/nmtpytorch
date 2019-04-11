# -*- coding: utf-8 -*-
import torch


class UniformAttention(torch.nn.Module):
    """A dummy non-parametric attention layer that applies uniform weights."""
    def __init__(self):
        super().__init__()

    def forward(self, hid, ctx, ctx_mask=None):
        alpha = torch.ones(*ctx.shape[:2], device=ctx.device).div(ctx.shape[0])
        wctx = (alpha.unsqueeze(-1) * ctx).sum(0)
        return alpha, wctx
