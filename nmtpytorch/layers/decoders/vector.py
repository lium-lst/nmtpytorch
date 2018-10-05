# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

from .. import FF

# Decoder without attention that uses a single input vector.
# Layer contributed by @loicbarrault


class VectorDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, ctx_size_dict, ctx_name, n_vocab,
                 tied_emb=False,
                 dropout_out=0,
                 emb_maxnorm=None, emb_gradscale=False):
        super().__init__()

        # Other arguments
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_size_dict = ctx_size_dict
        self.ctx_name = ctx_name
        self.n_vocab = n_vocab
        self.tied_emb = tied_emb
        self.dropout_out = dropout_out
        self.emb_maxnorm = emb_maxnorm
        self.emb_gradscale = emb_gradscale

        # Create target embeddings
        self.emb = nn.Embedding(self.n_vocab, self.input_size,
                                padding_idx=0, max_norm=self.emb_maxnorm,
                                scale_grad_by_freq=self.emb_gradscale)

        # Create decoder layer
        self.dec = nn.GRUCell(self.input_size, self.hidden_size)

        self.ff_dec_init = FF(
            self.ctx_size_dict[self.ctx_name],
            self.hidden_size, activ='tanh')

        # Output dropout
        if self.dropout_out > 0:
            self.do_out = nn.Dropout(p=self.dropout_out)

        # Output bottleneck: maps hidden states to target emb dim
        self.hid2out = FF(self.hidden_size,
                          self.input_size, bias_zero=True, activ='tanh')

        # Final softmax
        self.out2prob = FF(self.input_size, self.n_vocab)

        # Tie input embedding matrix and output embedding matrix
        if self.tied_emb:
            self.out2prob.weight = self.emb.weight

        self.nll_loss = nn.NLLLoss(reduction="sum", ignore_index=0)

    def f_init(self, ctx_dict):
        """Returns the initial h_0 for the decoder."""
        # unpack the context
        ctx, ctx_mask = ctx_dict[self.ctx_name]
        assert ctx_mask is None, "Mask is not None"

        # initialize with source ctx
        return self.ff_dec_init(ctx)

    def f_next(self, ctx_dict, y, h):
        h1 = self.dec(y, h)

        # no more h2 since we only have 1 GRU

        # hidden -> inp
        self.h3 = self.hid2out(h1)

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
