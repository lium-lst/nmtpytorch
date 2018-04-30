# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FF(nn.Module):
    """A smart feedforward layer with activation support."""
    def __init__(self, in_features, out_features, bias=True,
                 bias_zero=True, activ=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_zero = bias_zero
        self.activ_type = activ if activ else 'linear'
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        if activ is None:
            self.activ = lambda x: x
        else:
            self.activ = getattr(F, activ)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            if self.bias_zero:
                self.bias.data.zero_()
            else:
                self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return self.activ(F.linear(input, self.weight, self.bias))

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', activ=' + str(self.activ_type) + ')'
