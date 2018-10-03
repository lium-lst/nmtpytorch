# -*- coding: utf-8 -*-
import torch
from torch import nn

from ..utils.nn import get_activation_fn
from . import FF


# TODO: allow for returning a sequence of z states (will require mask)


class ZSpace(nn.Module):
    """Latent "z" space for combining the results of multiple encoders
        in a multitask setup.

    Arguments:
        ctx_size_dict (dict): Dictionary with key-value pairs {encoder_type : input_size}
        z_size (int): Size of the z-space vectors.
        z_type (str, optional): Design of network, i.e. single-layer, multi-
            layer, highway (Default: 'ff' i.e. single-layer feed-forward).
        activ (str, optional): Nonlinearity applied to the z-space (Default: None).

    Input:
        x (dict): Dictionary of encoder results with key-value pairs
            {modality : encoder_result}.

    Output:
        z (Tensor): A single-dimensional tensor of shape z_size.
    """
    def __init__(self, ctx_size_dict, z_size, z_type=None, activ=None):
        super().__init__()
        self.ctx_size_dict = ctx_size_dict
        self.z_size = z_size
        self.z_type = z_type.lower() if z_type else None
        self.activ = get_activation_fn(activ)

        # Safety check
        assert self.z_type in (None, 'ff', 'multi', 'highway'), \
            "layer z_type '{}' not known".format(z_type)

        if self.z_type is None:
            assert(len(set([size for size in ctx_size_dict.values()])) == 1), \
                "Encoder vector sizes are not equal! Consider using z_type:ff in config."
        elif z_type == 'ff':
            self.z_proj = nn.ModuleDict()
            for k, v in self.ctx_size_dict.items():
                self.z_proj[k] = FF(v, self.z_size, activ=None)
        else:
            raise Exception('z_type other than None or FF are not implemented yet.')

        self.setup_forward()

    def setup_forward(self):
        """ Sets up specified network architecture """
        if self.z_type is None:
            self.forward = self.forward_none
        elif self.z_type == 'ff':
            self.forward = self.forward_ff

    def forward_none(self, x):
        """ Single layer feed-forward layer
            x is a dict of tuples (values, mask), simply operate on values
        """
        # if encoder results are different lengths, first project to z_size
        comb = torch.cat([x[k][0] for k in x], 0)
        self.combination = torch.sum(comb, 0)
        return self.combination

    def forward_ff(self, x):
        """ Single layer feed-forward layer
            x is a dict of tuples (values, mask), simply operate on values
        """
        # first project to z_size

        # FIXME: This is not working
        projectors = nn.ModuleDict()
        for modality, enc_vec in x.items():
            projectors[modality] = FF(
                enc_vec[0].shape[0], self.z_size, activ=None)

        return self.activ(self.combination)
        # TODO: get parameter from multitask.py to project to z_size which != enc_size
        # return self.combination
