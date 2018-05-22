# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from ..utils.data import to_var
from . import FF


class DecoderInitializer(nn.Module):
    """Decoder initializer block for GRU and LSTM.

    Arguments:
        dec_type(str): GRU or LSTM.
        input_size(int): Input dimensionality of the feature vectors that'll
            be used for initialization if ``method != zero``.
        hidden_size(int): Output dimensionality, i.e. hidden size of the decoder
            that will be initialized.
        data_source(str): The modality name to look for in the batch dictionary.
        method(str): One of ``last_ctx|mean_ctx|feats|zero``.
        activ(str, optional): The non-linearity to be used for all initializers
            except 'zero'. Default is ``None`` i.e. no non-linearity.
    """
    def __init__(self, dec_type, input_size, hidden_size, data_source,
                 method, activ=None):
        self.dec_type = dec_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.data_source = data_source
        self.method = method
        self.activ = activ

        # Check for decoder type
        assert self.dec_type in ('GRU', 'LSTM'), \
            "dec_type '{}' is unknown.".format(self.dec_type)

        assert self.method in ('mean_ctx', 'last_ctx', 'zero', 'feats'), \
            "Decoder init method '{}' is unknown.".format(self.method)

        # LSTMs have also the cell state so double the output size
        self.n_states = 1 if self.dec_type == 'GRU' else 2

        if self.method in ('mean_ctx', 'last_ctx', 'feats'):
            self.ff = FF(
                self.input_size, self.hidden_size * self.n_states,
                activ=self.activ)

        # Set the actual initializer depending on the method
        self._initializer = getattr(self, '_init_{}'.format(self.method))

    def forward(self, ctx_dict):
        ctx, ctx_mask = ctx_dict[self.data_source]
        return self._initializer(ctx, ctx_mask)

    def _init_zero(self, ctx, mask):
        h_0 = torch.zeros(ctx.shape[1], self.hidden_size * self.n_states)
        return to_var(h_0, requires_grad=False)

    def _init_feats(self, ctx, mask):
        return self.ff(ctx)

    def _init_mean_ctx(self, ctx, mask):
        if mask is None:
            return self.ff(ctx.mean(0))
        else:
            return self.ff(ctx.sum(0) / mask.sum(0).unsqueeze(1))

    def _init_last_ctx(self, ctx, mask):
        if mask is None:
            return self.ff(ctx[-1])
        else:
            # Fetch last timesteps
            last_tsteps = mask.sum(0).sub(1).long()
            return self.ff(ctx[last_tsteps, range(ctx.shape[1])])

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features={}'.format(self.input_size) \
            + 'out_features={}'.format(self.hidden_size) \
            + 'activ={}'.format(self.activ) \
            + 'method={}'.format(self.method) + ')'
