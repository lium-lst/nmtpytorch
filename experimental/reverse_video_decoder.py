# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.autograd import Variable

from ..utils.nn import get_rnn_hidden_state
from . import FF, Attention


class ReverseVideoDecoder(nn.Module):
    """
        A reverse video feature reconstruction decoder
    """
    def __init__(self, input_size, hidden_size, ctx_size_dict, ctx_name,
                 rnn_type, video_dim, dec_init='zero',
                 dec_init_size=None, att_type='mlp',
                 att_activ='tanh', att_bottleneck='ctx', att_temp=1.0,
                 use_smoothL1=False, transform_ctx=True, mlp_bias=False, dropout_out=0):

        super().__init__()

        # Normalize case
        self.rnn_type = rnn_type.upper()

        # Safety checks
        assert self.rnn_type in ('GRU', 'LSTM'), \
            "rnn_type '{}' not known".format(rnn_type)
        assert dec_init in ('zero', 'mean_ctx', 'feats'), \
            "dec_init '{}' not known".format(dec_init)

        RNN = getattr(nn, '{}Cell'.format(self.rnn_type))
        # LSTMs have also the cell state
        self.n_states = 1 if self.rnn_type == 'GRU' else 2

        # Set custom handlers for GRU/LSTM
        if self.rnn_type == 'GRU':
            self._rnn_unpack_states = lambda x: x
            self._rnn_pack_states = lambda x: x
        elif self.rnn_type == 'LSTM':
            self._rnn_unpack_states = self._lstm_unpack_states
            self._rnn_pack_states = self._lstm_pack_states

        # Set decoder initializer
        self._init_func = getattr(self, '_rnn_init_{}'.format(dec_init))

        # Other arguments
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_size_dict = ctx_size_dict
        self.ctx_name = ctx_name
        self.dec_init = dec_init
        self.dec_init_size = dec_init_size
        self.att_type = att_type
        self.att_bottleneck = att_bottleneck
        self.att_activ = att_activ
        self.att_temp = att_temp
        self.transform_ctx = transform_ctx
        self.mlp_bias = mlp_bias
        self.dropout_out = dropout_out
        self.video_dim = video_dim
        self.use_smoothL1 = use_smoothL1

        # Create a video frame embedding layer that maps the video_dim(2048) -> proj_size
        self.emb = FF(self.video_dim, self.hidden_size, bias=True, activ='tanh')

        # Create an attention layer
        self.att = Attention(self.ctx_size_dict[self.ctx_name], self.hidden_size,
                             transform_ctx=self.transform_ctx,
                             mlp_bias=self.mlp_bias,
                             att_type=self.att_type,
                             att_activ=self.att_activ,
                             att_bottleneck=self.att_bottleneck,
                             temp=self.att_temp)

        # Decoder initializer FF (for 'mean_ctx' or auxiliary 'feats')
        if self.dec_init in ('mean_ctx', 'feats'):
            if self.dec_init == 'mean_ctx':
                self.dec_init_size = self.ctx_size_dict[self.ctx_name]
            self.ff_dec_init = FF(
                self.dec_init_size,
                self.hidden_size * self.n_states, activ='tanh')

        # Create first decoder layer necessary for attention
        self.dec0 = RNN(self.hidden_size, self.hidden_size)
        self.dec1 = RNN(self.hidden_size, self.hidden_size)

        # Output dropout
        if self.dropout_out > 0:
            self.do_out = nn.Dropout(p=self.dropout_out)

        # Output bottleneck: maps hidden states to target emb dim
        self.hid2out = FF(self.hidden_size, self.video_dim)

        #  MSE loss
        self.MSE_loss = nn.MSELoss(size_average=False)
        #  SmoothL1 loss
        self.SmoothL1_loss = nn.SmoothL1Loss(size_average=False)

    def _lstm_pack_states(self, h):
        return torch.cat(h, dim=-1)

    def _lstm_unpack_states(self, h):
        # Split h_t and c_t into two tensors and return a tuple
        return torch.split(h, self.hidden_size, dim=-1)

    def _rnn_init_zero(self, ctx_dict):
        ctx, _ = ctx_dict[self.ctx_name]
        h_0 = torch.zeros(ctx.shape[1], self.hidden_size * self.n_states)
        return Variable(h_0).cuda()

    def _rnn_init_mean_ctx(self, ctx_dict):
        ctx, ctx_mask = ctx_dict[self.ctx_name]
        if ctx_mask is None:
            return self.ff_dec_init(ctx.mean(0))
        else:
            return self.ff_dec_init(ctx.sum(0) / ctx_mask.sum(0).unsqueeze(1))

    def _rnn_init_feats(self, ctx_dict):
        ctx, _ = ctx_dict['feats']
        return self.ff_dec_init(ctx)

    def f_init(self, ctx_dict):
        """Returns the initial h_0 for the decoder."""
        self.alphas = []
        return self._init_func(ctx_dict)

    def f_next(self, ctx_dict, y, hidden):
        '''
        Encode the video frame features in the first layer of the RNN (dec0).
        Build the hidden layer

        '''
        # Get hidden states from the first decoder (purely cond. on LM)
        h1_c1 = self.dec0(y, self._rnn_unpack_states(hidden))
        h1 = get_rnn_hidden_state(h1_c1)

        # Apply attention
        self.txt_alpha_t, txt_z_t = self.att(
            h1.unsqueeze(0), *ctx_dict[self.ctx_name])

        # Run second decoder (h1 is compatible now as it was returned by GRU)
        h2_c2 = self.dec1(txt_z_t, h1_c1)
        h2 = get_rnn_hidden_state(h2_c2)

        # This is a bottleneck to avoid going from H to V directly
        predicted_v = self.hid2out(h2)

        return predicted_v, self._rnn_pack_states(h2_c2)

    def forward(self, ctx_dict, y, use_smoothL1=False):
        """
                Computes the Mean Squared Error or the SmoothL1 between the predicted
        feature vector and the ground-truth video features indices `y`.
        Only called during training.

        """

        loss = 0.0

        # reverse the order of the target sequence
        y = y.transpose(0, 1)
        idx = [i for i in range(y.size(1) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).cuda()
        y = y.index_select(1, idx).transpose(0, 1)

        # Get initial hidden state
        hidden = self.f_init(ctx_dict)
        # Initializing the first step of the decoder with zeros
        predicted_v = Variable(torch.zeros(y.shape[1], self.video_dim)).cuda()

        for t in range(y.shape[0]):
            predicted_v = self.emb(predicted_v)
            predicted_v, hidden = self.f_next(ctx_dict, predicted_v, hidden)
            if use_smoothL1:
                loss += self.SmoothL1_loss(predicted_v, y[t])
            else:
                loss += self.MSE_loss(predicted_v, y[t])

        return {'loss': loss, 'logps': None, 'n_items': y.shape[0] * y.shape[1]}
