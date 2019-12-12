# -*- coding: utf-8 -*-
import math

import torch


class ScaledDotAttention(torch.nn.Module):
    """Scaled Dot-product attention from `Attention is all you need`.

    Arguments:

    Input:

    Output:
    """

    def __init__(self, model_dim, n_heads, causal=False):
        super().__init__()
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.causal = causal

        #self.k_dim = self.model_dim / self.n_heads
        #self.v_dim = self.model_dim / self.n_heads

        # Efficient linear projections for all heads
        self.lin_k = torch.nn.Linear(
            self.model_dim, self.model_dim, bias=False)
        self.lin_q = torch.nn.Linear(
            self.model_dim, self.model_dim, bias=False)
        self.lin_v = torch.nn.Linear(
            self.model_dim, self.model_dim, bias=False)

        # Final output layer is independent of number of heads
        self.lin_o = torch.nn.Linear(
            self.model_dim, self.model_dim, bias=False)

        self.scale = math.sqrt(self.model_dim / self.n_heads)

    def forward(self, inputs):
        """Scaled dot-product attention forward-pass

        :param inputs: dictionary with query, key, value and mask tensors
            the shape of the tensors are (tstep, bsize, dim) except for the
            mask which is (tstep, bsize)

        :return: foo
        """
        q, k, v, mask = inputs
        # q is the query, v is the actual inputs and k is v's representation
        # for self attention q=v=k
        # for cross attention q != (v=k)
        # Project keys, queries and values --> (bsize, tstep, dim)
        tstep, bsize = mask.shape
        head_view = (tstep, bsize, self.n_heads, -1)
        # qp: (bsize, head, tstep, dim)
        # vp: (bsize, head, tstep, dim)
        # kp: (bsize, head, dim, tstep)
        qp = self.lin_q(q).view(*head_view).permute(1, 2, 0, 3)
        vp = self.lin_v(v).view(*head_view).permute(1, 2, 0, 3)
        kp = self.lin_k(k).view(*head_view).permute(1, 2, 3, 0)

        # z: (bsize, head, tstep, tstep)
        z = torch.matmul(qp, kp).div(self.scale).softmax(dim=-1)
        out = torch.matmul(z, vp).permute(2, 0, 1, 3).reshape_as(v)
        return (v, out, mask)
