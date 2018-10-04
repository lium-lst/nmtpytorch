# -*- coding: utf-8 -*-
import torch
from torch import nn

from . import FF, HierarchicalAttention, get_attention
from ..utils.device import DEVICE


# TODO: allow for returning a sequence of z states (will require mask)


class ZSpaceAtt(nn.Module):
    """Latent "z" space for combining the results of multiple encoders
        in a multitask setup. This is done by attending to each input modality
        and generate a fixed-length sequence of states.

    Arguments:
        ctx_size_dict (dict): Dictionary with key-value pairs {encoder_type : input_size}
        z_size (int): Size of the z-space vectors.
        z_len (int, optional): Length of the z-space sequence (Default: 10).
        z_transform: (str, optional): How should the contexts be transformed
            to match z_in_size? [None|linear|tanh|sigmoid] (Default: None)
        z_in_size: (int, optional): input size of the z RNN layer (Default: 256).
        z_merge: (str, optional): How to merge the attended vectors
            from each modality? [sum|hierarchical] (Default: sum).
        z_init (str, optional): How to initialize the Z space [sum|mean] (Default: sum).
        att_type (str, optional): Attention type [mlp|dot] (Default: mlp).
        att_activ (str, optional): Activation function for attention
            [linear|sigmoid|tanh] (Default: tanh)
        att_bottleneck (str, optional): Attention mechanism hidden size
            [ctx|hid|<integer>] (Default: ctx).
        att_temp (float, optional): Attention temperature (Default: 1.0).
        att_transform_ctx (bool, optional): Should we transform ctx for
            attention mechanism (Default: False).
        mlp_bias (bool, optional): Whether attention mechanism should
            incorporate a bias or no (Default: False).
        hiero_mid_dim(int, optional): Common dimensionality for hierarchical
            attention (Default: 128).

    Input:
        x (dict): Dictionary of encoder results with key-value pairs
            {modality : encoder_result}.

    Output:
        z (Tensor): A sequence of z_len-dimensional vectors of shape z_size.
    """

    def __init__(self, ctx_size_dict, z_size, z_len=10, z_transform=None, z_in_size=256,
                 z_merge='sum', z_init='mean_ctx', att_type='mlp',
                 att_activ='tanh', att_bottleneck='ctx', att_temp=1.0,
                 att_transform_ctx=False, mlp_bias=False, hiero_mid_dim=128):
        super().__init__()

        self.ctx_size_dict = ctx_size_dict
        self.z_size = z_size
        self.z_len = z_len
        self.z_transform = z_transform.lower() if z_transform else None
        self.z_in_size = z_in_size
        self.z_merge = z_merge

        # Other arguments
        self.att_type = att_type
        self.att_bottleneck = att_bottleneck
        self.att_activ = att_activ
        self.att_temp = att_temp
        self.att_transform_ctx = att_transform_ctx
        self.mlp_bias = mlp_bias
        self.z_init = z_init
        self.hiero_mid_dim = hiero_mid_dim

        # Safety check
        self._sanity_check()

        # Create FF layers to manage different context size...
        # Each layer maps ctx_size_dict[k] to z_in_size (==ctx_size)
        # z_transform tells the kind of (non-)linearity to use
        if self.z_transform:
            self.z_transforms = nn.ModuleDict()
            for k in self.ctx_size_dict:
                self.z_transforms[k] = FF(
                    self.ctx_size_dict[k], self.z_in_size, activ=z_transform)
                self.ctx_size_dict[k] = self.z_in_size
            self.ctx_size = self.z_in_size
        else:
            s = set([size for size in self.ctx_size_dict.values()])
            assert len(set(s)) == 1, \
                "Incompatible encoding sizes, consider using z_transform:tanh in config."
            self.ctx_size = next(iter(s))

        # Create an attention layer for each modality
        # TODO: sharing weights between att. mechanisms is possible
        self.att = nn.ModuleDict()
        # Fetch correct attention class
        Attention = get_attention(self.att_type)
        for k in self.ctx_size_dict:
            att_in_size = self.ctx_size if self.z_transform else self.ctx_size_dict[k]
            self.att[k] = Attention(
                att_in_size, self.z_size,
                transform_ctx=self.att_transform_ctx,
                mlp_bias=self.mlp_bias,
                att_activ=self.att_activ,
                att_bottleneck=self.att_bottleneck,
                temp=self.att_temp,
                ctx2hid=False)

        # Fusion operation
        if self.z_merge == 'hierarchical':
            self.hiero_att = HierarchicalAttention(
                [self.ctx_size_dict[k] for k in self.ctx_size_dict.keys()],
                self.z_size, self.hiero_mid_dim)
            self.merge_op = self._merge_hierarchical
        else:
            self.merge_op = self._merge_sum

        # Create decoder layer necessary for attention
        self.dec = nn.GRUCell(self.ctx_size, self.z_size)

        # Several strategies to initialize the decoder can be considered
        # Set decoder initializer
        self._init_func = getattr(self, '_rnn_init_{}'.format(self.z_init))

        # if init is not zero, then create FF layer
        if self.z_init != 'zero':
            self.ff_z_init = FF(
                self.ctx_size,
                self.z_size, activ='tanh')

    def _sanity_check(self):
        assert self.z_transform in (None, 'linear', 'tanh', 'sigmoid'), \
            "layer z_transform '{}' not known".format(self.z_transform)
        assert self.z_init in ('zero', 'mean_ctx'), \
            "z_init '{}' not known".format(self.z_init)

    def _rnn_init_zero(self, ctx_dict):
        # * self.n_states) # <-- ?? was used in cond_decoder.py
        return torch.zeros(self.ctx_size, self.z_size, device=DEVICE)

    def _rnn_init_mean_ctx(self, ctx_dict):
        # NOTE: averaging the mean of all modalities
        # NOTE: all ctx should have the same size at this point
        key = next(iter(ctx_dict))
        res = torch.zeros(ctx_dict[key][0].shape[1:], device=DEVICE)
        for e in ctx_dict.keys():
            ctx, ctx_mask = ctx_dict[e]
            if ctx_mask is None:
                res += ctx.mean(0)
            else:
                res += ctx.sum(0) / ctx_mask.sum(0).unsqueeze(1)
        return self.ff_z_init(res / len(ctx_dict))

    def f_init(self, ctx_dict):
        """Returns the initial h_0 for the decoder."""
        self.alphas = []
        return self._init_func(ctx_dict)

    def forward(self, ctx_dict):
        """ctx_dict is a dict of tuples (values, mask), simply operate on values."""
        z_states_list = []
        # Transform contexts if needed
        if self.z_transform:
            trans_ctx_dict = {}
            for k in ctx_dict:
                trans_ctx_dict[k] = (self.z_transforms[k](ctx_dict[k][0]), ctx_dict[k][1])
            ctx_dict = trans_ctx_dict

        # Get initial hidden state
        h = self.f_init(ctx_dict)

        # loop for z_len timesteps
        for t in range(self.z_len):
            # compute attended vector for each encoders
            self.alpha_t = {}
            z_t = {}
            for e in ctx_dict:
                # Apply attention
                self.alpha_t[e], z_t[e] = self.att[e](
                    h.unsqueeze(0), *ctx_dict[e])
            # merge all attended vectors
            self.hiero_att_weights, fusion = self.merge_op(z_t, h)
            # feed the decoder RNN with the result
            h = self.dec(fusion, h)
            z_states_list.append(h.unsqueeze(0))

        # store the states into a tensor so that the decoders can use it seemlessly
        z_states = torch.cat(z_states_list, 0)
        return z_states

    # NOTE: h is never used, it's for compatibility with _merge_hierarchical
    def _merge_sum(self, att_ctx_dict, h):
        summ = None
        for e in att_ctx_dict.keys():
            if summ is None:
                summ = torch.zeros(att_ctx_dict[e].shape, device=DEVICE)
            summ += att_ctx_dict[e]
        return None, summ

    def _merge_hierarchical(self, att_ctx_dict, h):
        att_ctx_list = [att_ctx_dict[k] for k in att_ctx_dict.keys()]
        return self.hiero_att(att_ctx_list, h)
