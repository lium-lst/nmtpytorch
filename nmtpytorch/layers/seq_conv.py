# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn


# Code contributed by @jlibovicky


class SequenceConvolution(nn.Module):
    """1D convolution with optional max-pooling.

    The layer applies 1D convolution of odd kernel size with output channel
    counts specified by a list of integers. Then, it optionally applies 1D
    max-pooling to reduce the sequence length.
    """

    def __init__(self, input_dim, filters, max_pool_stride=None, activation='relu'):
        super().__init__()
        self.max_pool_stride = max_pool_stride

        self.conv_proj = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim,
                      out_channels=size,
                      kernel_size=2 * k + 1,
                      padding=k)
            for k, size in enumerate(filters) if size > 0])

        if self.max_pool_stride is not None:
            self.max_pool = nn.MaxPool1d(
                kernel_size=self.max_pool_stride,
                stride=self.max_pool_stride)
        else:
            self.max_pool = None

    def forward(self, x, mask):
        conv_outputs = [conv(x.permute(1, 2, 0)) for conv in self.conv_proj]
        conv_out = torch.cat(conv_outputs, dim=1)

        if self.max_pool is not None:
            conv_len = conv_out.size(-1)
            if conv_len < self.max_pool_stride:
                pad_size = self.max_pool_stride - conv_len
                conv_out = F.pad(conv_out, pad=[pad_size, pad_size])
            max_pooled_data = self.max_pool(conv_out).permute(2, 0, 1)
            max_pooled_mask = (self.max_pool(mask.t().unsqueeze(1)).squeeze(1).t()
                               if mask is not None else None)
            return max_pooled_data, max_pooled_mask
        else:
            return conv_out.permute(2, 0, 1), mask
