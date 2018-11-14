# -*- coding: utf-8 -*-
from collections import defaultdict
import torch
from torch import nn
import torch.nn.functional as F

from ...utils.nn import get_rnn_hidden_state
from .. import FF
from ..attention import get_attention


class XuDecoder(nn.Module):
    """A decoder which implements Show-attend-and-tell decoder."""
    def __init__(self, input_size, hidden_size, ctx_size_dict, ctx_name, n_vocab,
                 rnn_type, tied_emb=False, dec_init='zero', att_type='mlp',
                 att_activ='tanh', att_bottleneck='ctx',
                 transform_ctx=True, mlp_bias=True, dropout=0,
                 emb_maxnorm=None, emb_gradscale=False, att_temp=1.0,
                 selector=False, prev2out=True, ctx2out=True):
        super().__init__()

        # Normalize case
        self.rnn_type = rnn_type.upper()

        # Safety checks
        assert self.rnn_type in ('GRU', 'LSTM'), \
            "rnn_type '{}' not known".format(rnn_type)
        assert dec_init in ('zero', 'mean_ctx'), \
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
        self.n_vocab = n_vocab
        self.dropout = dropout
        self.ctx2out = ctx2out
        self.selector = selector
        self.prev2out = prev2out
        self.tied_emb = tied_emb
        self.dec_init = dec_init
        self.ctx_name = ctx_name
        self.mlp_bias = mlp_bias
        self.att_temp = att_temp
        self.att_activ = att_activ
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.emb_maxnorm = emb_maxnorm
        self.emb_gradscale = emb_gradscale
        self.transform_ctx = transform_ctx
        self.ctx_size_dict = ctx_size_dict
        self.att_bottleneck = att_bottleneck

        # Create target embeddings
        self.emb = nn.Embedding(self.n_vocab, self.input_size,
                                padding_idx=0, max_norm=self.emb_maxnorm,
                                scale_grad_by_freq=self.emb_gradscale)

        # Create attention layer
        Attention = get_attention(att_type)
        self.att = Attention(self.ctx_size_dict[self.ctx_name], self.hidden_size,
                             transform_ctx=self.transform_ctx,
                             mlp_bias=self.mlp_bias,
                             att_activ=self.att_activ,
                             att_bottleneck=self.att_bottleneck,
                             temp=self.att_temp, ctx2hid=False)

        # Decoder initializer FF (for mean_ctx)
        if self.dec_init == 'mean_ctx':
            self.ff_dec_init = FF(
                self.ctx_size_dict[self.ctx_name],
                self.hidden_size * self.n_states, activ='tanh')

        # Dropout
        if self.dropout > 0:
            self.do = nn.Dropout(p=self.dropout)

        # Gating Scalar, i.e. selector
        if self.selector:
            self.ff_selector = FF(self.hidden_size, 1, activ='sigmoid')

        if self.ctx2out:
            self.ff_out_ctx = FF(
                self.ctx_size_dict[self.ctx_name], self.input_size)

        # Create decoder from [y_t, z_t] to dec_dim
        self.dec0 = RNN(
            self.input_size + self.ctx_size_dict[self.ctx_name],
            self.hidden_size)

        # Output bottleneck: maps hidden states to target emb dim
        self.hid2out = FF(self.hidden_size, self.input_size)

        # Final softmax
        self.out2prob = FF(self.input_size, self.n_vocab)

        # Tie input embedding matrix and output embedding matrix
        if self.tied_emb:
            self.out2prob.weight = self.emb.weight

        self.nll_loss = nn.NLLLoss(reduction="sum", ignore_index=0)

    def _lstm_pack_states(self, h):
        return torch.cat(h, dim=-1)

    def _lstm_unpack_states(self, h):
        # Split h_t and c_t into two tensors and return a tuple
        return torch.split(h, self.hidden_size, dim=-1)

    def _rnn_init_zero(self, ctx, ctx_mask):
        return torch.zeros(
            ctx.shape[1], self.hidden_size * self.n_states, device=ctx.device)

    def _rnn_init_mean_ctx(self, ctx, ctx_mask):
        mean_ctx = ctx.mean(dim=0)
        if self.dropout > 0:
            mean_ctx = self.do(mean_ctx)
        return self.ff_dec_init(mean_ctx)

    def f_init(self, ctx_dict):
        """Returns the initial h_0, c_0 for the decoder."""
        self.history = defaultdict(list)
        return self._init_func(*ctx_dict[self.ctx_name])

    def f_next(self, ctx_dict, y, h):
        # Unpack hidden states
        h_c = self._rnn_unpack_states(h)

        # Apply attention
        img_alpha_t, z_t = self.att(
            h_c[0].unsqueeze(0), *ctx_dict[self.ctx_name])
        # Save reg loss terms
        self.history['alpha_img'].append(img_alpha_t.unsqueeze(0))

        if self.selector:
            z_t *= self.ff_selector(h_c[0])

        # Form RNN input by concatenating embedding and weighted sum
        # Give h as dec's hidden_state
        ht_ct = self.dec0(torch.cat([y, z_t], dim=1), h_c)

        # Get h_t from the combined ht_ct vector
        h_t = get_rnn_hidden_state(ht_ct)
        if self.dropout > 0:
            h_t = self.do(h_t)

        # This h_t, (optionally along with y and z_t)
        # will connect to softmax() predictions.
        logit = self.hid2out(h_t)

        if self.prev2out:
            logit += y

        if self.ctx2out:
            logit += self.ff_out_ctx(z_t)

        logit = torch.tanh(logit)
        if self.dropout > 0:
            logit = self.do(logit)

        # Transform logit to T*B*V (V: vocab_size)
        # Compute log_softmax over token dim
        log_p = F.log_softmax(self.out2prob(logit), dim=-1)

        # Return log probs and new hidden states
        return log_p, self._rnn_pack_states(ht_ct)

    def forward(self, ctx_dict, y):
        loss = 0.0
        logps = None if self.training else torch.zeros(
            y.shape[0] - 1, y.shape[1], self.n_vocab, device=y.device)

        # Convert token indices to embeddings -> T*B*E
        y_emb = self.emb(y)

        # Get initial hidden state
        h = self.f_init(ctx_dict)

        # -1: So that we skip the timestep where input is <eos>
        for t in range(y_emb.shape[0] - 1):
            log_p, h = self.f_next(ctx_dict, y_emb[t], h)
            if not self.training:
                logps[t] = log_p.data
            loss += self.nll_loss(log_p, y[t + 1])

        return {'loss': loss, 'logps': logps}
