# -*- coding: utf-8 -*-
import operator
from functools import reduce

import torch

from . import FF


class Fusion(torch.nn.Module):
    """A convenience fusion layer that merges an arbitrary number of inputs.

    Arguments:
        fusion_type(str, optional): One of ``concat,sum,mul`` defining the
            fusion operation. In the default setup of ``concat``, the
            following two arguments should be provided to create a
            ``Linear`` adaptor which will project the concatenated vector to
            ``output_size``.
        input_size(int, optional): The dimensionality of the concatenated
            input. Only necessary if ``fusion_type==concat``.
        output_size(int, optional): The output dimensionality of the
            concatenation. Only necessary if ``fusion_type==concat``.
    """

    def __init__(self, fusion_type='concat', input_size=None, output_size=None):
        super().__init__()

        self.fusion_type = fusion_type
        self.forward = getattr(self, '_{}'.format(self.fusion_type))

        if self.fusion_type == 'concat':
            assert input_size and output_size, \
                "input_size and output_size should be given for concat"
            self.adaptor = FF(input_size, output_size, bias=False, activ=None)

    def _sum(self, *inputs):
        return reduce(operator.add, inputs)

    def _mul(self, *inputs):
        return reduce(operator.mul, inputs)

    def _concat(self, *inputs):
        return self.adaptor(torch.cat(inputs, dim=-1))

    def __repr__(self):
        return "Fusion(type={}, adaptor={})".format(
            self.fusion_type,
            getattr(self, 'adaptor') if hasattr(self, 'adaptor') else 'None')
