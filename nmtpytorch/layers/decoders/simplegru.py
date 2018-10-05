# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

from .. import FF
from ..attention import get_attention


class SimpleGRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, ctx_size_dict, ctx_name, n_vocab,
                 tied_emb=False, dec_init='mean_ctx', att_type='mlp',
                 att_activ='tanh', att_bottleneck='ctx', att_temp=1.0,
                 transform_ctx=True, mlp_bias=False, dropout_out=0,
                 emb_maxnorm=None, emb_gradscale=False):
        super().__init__()

        # Other arguments
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_size_dict = ctx_size_dict
        self.ctx_name = ctx_name
        self.n_vocab = n_vocab
        self.tied_emb = tied_emb
        self.att_type = att_type
        self.att_bottleneck = att_bottleneck
        self.att_activ = att_activ
        self.att_temp = att_temp
        self.transform_ctx = transform_ctx
        self.mlp_bias = mlp_bias
        self.dropout_out = dropout_out
        self.emb_maxnorm = emb_maxnorm
        self.emb_gradscale = emb_gradscale
        self.dec_init = dec_init

        # Create target embeddings
        self.emb = nn.Embedding(self.n_vocab, self.input_size,
                                padding_idx=0, max_norm=self.emb_maxnorm,
                                scale_grad_by_freq=self.emb_gradscale)

        # Create attention layer
        Attention = get_attention(self.att_type)
        self.att = Attention(self.ctx_size_dict[self.ctx_name], self.hidden_size,
                             transform_ctx=self.transform_ctx,
                             mlp_bias=self.mlp_bias,
                             att_activ=self.att_activ,
                             att_bottleneck=self.att_bottleneck,
                             temp=self.att_temp)

        # Create first decoder layer necessary for attention
        self.dec0 = nn.GRUCell(self.input_size, self.hidden_size)

        if self.dec_init != 'zero':
            self.ff_dec_init = FF(
                self.ctx_size_dict[self.ctx_name],
                self.hidden_size, activ='tanh')

        # Output dropout
        if self.dropout_out > 0:
            self.do_out = nn.Dropout(p=self.dropout_out)

        # Output bottleneck: maps hidden states to target emb dim
        self.hid2out = FF(2 * self.hidden_size,
                          self.input_size, bias_zero=True, activ='tanh')

        # Final softmax
        self.out2prob = FF(self.input_size, self.n_vocab)

        # Tie input embedding matrix and output embedding matrix
        if self.tied_emb:
            self.out2prob.weight = self.emb.weight

        self.nll_loss = nn.NLLLoss(reduction="sum", ignore_index=0)

    def f_init(self, ctx_dict):
        """Returns the initial h_0 for the decoder."""
        # reset attentional state
        self.h3 = None

        # unpack the context
        ctx, ctx_mask = ctx_dict[self.ctx_name]

        if self.dec_init == 'zero':
            return torch.zeros(ctx.shape[1], self.hidden_size, device=ctx.device)

        elif self.dec_init == 'mean_ctx':
            h_0 = self.ff_dec_init(
                ctx.sum(0).div(ctx_mask.unsqueeze(-1).sum(0))
                if ctx_mask is not None else ctx.mean(0))
        elif self.dec_init == 'max':
            h_0 = self.ff_dec_init(ctx.max(0)[0])
        elif self.dec_init == 'last':
            if ctx_mask is None:
                h_0 = self.ff_dec_init(ctx[-1])
            else:
                last_idxs = ctx_mask.sum(0).sub(1).long()
                h_0 = self.ff_dec_init(ctx[last_idxs, range(ctx.shape[1])])

        # return initial context
        return h_0

    def f_next(self, ctx_dict, y, h):
        h1 = self.dec0(y, h)

        # Apply attention
        self.alpha_t, z_t = self.att(h1.unsqueeze(0), *ctx_dict[self.ctx_name])

        # Concatenate attented source and hidden state
        h2 = torch.cat((h1, z_t), dim=-1)

        # hidden -> inp
        self.h3 = self.hid2out(h2)

        # Apply dropout if any
        if self.dropout_out > 0:
            logit = self.do_out(self.h3)
        else:
            logit = self.h3

        # Transform logit to T*B*V (V: vocab_size)
        # Compute log_softmax over token dim
        log_p = F.log_softmax(self.out2prob(logit), dim=-1)

        # Return log probs and new hidden states
        return log_p, h1

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
