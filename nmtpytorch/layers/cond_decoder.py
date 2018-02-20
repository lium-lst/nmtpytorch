# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.data import to_var
from . import FF, Attention


class ConditionalDecoder(nn.Module):
    """A conditional decoder with attention à la dl4mt-tutorial."""
    def __init__(self, input_size, hidden_size, ctx_size, n_vocab,
                 tied_emb=False, dec_init='zero', att_type='bahdanau',
                 dropout_out=0):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        self.n_vocab = n_vocab
        self.tied_emb = tied_emb
        self.dec_init = dec_init
        self.att_type = att_type
        self.dropout_out = dropout_out

        # Create target embeddings
        self.emb = nn.Embedding(self.n_vocab, self.input_size, padding_idx=0)

        # Create attention layer
        self.att = Attention(self.ctx_size, self.hidden_size,
                             att_type=self.att_type)

        # Decoder initializer
        if self.dec_init == 'mean_ctx':
            self.ff_dec_init = FF(self.ctx_size,
                                  self.hidden_size, activ='tanh')

        # Create first decoder layer necessary for attention
        self.dec0 = nn.GRUCell(self.input_size, self.hidden_size)
        self.dec1 = nn.GRUCell(self.hidden_size, self.hidden_size)

        # Output dropout
        if self.dropout_out > 0:
            self.do_out = nn.Dropout(p=self.dropout_out)

        # Output bottleneck: maps hidden states to target emb dim
        self.hid2out = FF(self.hidden_size, self.input_size,
                          bias_zero=True, activ='tanh')

        # Final softmax
        self.out2prob = FF(self.input_size, self.n_vocab)

        # Tie input embedding matrix and output embedding matrix
        if self.tied_emb:
            self.out2prob.weight = self.emb.weight

    def f_init(self, ctx, ctx_mask):
        """Returns the initial h_0 for the decoder."""
        # S*B*H
        if self.dec_init == 'zero':
            h_0 = torch.zeros(ctx.shape[1], self.hidden_size)
            return to_var(h_0, requires_grad=False)
        elif self.dec_init == 'mean_ctx':
            # Filter out zero positions
            return self.ff_dec_init(ctx.sum(0) / ctx_mask.sum(0).unsqueeze(1))

    def f_next(self, ctx_dict, y, h):
        # Get hidden states from the first decoder (purely cond. on LM)
        h1 = self.dec0(y, h)

        # Apply attention
        alpha_t, z_t = self.att(h1.unsqueeze(0), *ctx_dict['txt'])
        h2 = self.dec1(z_t, h1)

        # This is a bottleneck to avoid going from H to V directly
        logit = self.hid2out(h2)

        # Apply dropout if any
        if self.dropout_out > 0:
            logit = self.do_out(logit)

        # Transform logit to T*B*V (V: vocab_size)
        # Compute log_softmax over token dim
        log_p = -F.log_softmax(self.out2prob(logit), dim=-1)

        # Return log probs and new hidden states
        return log_p, h2

    def forward(self, ctx_dict, y):
        """Computes the softmax outputs given source annotations `ctxs` and
        ground-truth target token indices `y`.

        Arguments:
            ctxs(Variable): A variable of `S*B*ctx_dim` representing the source
                annotations in an order compatible with ground-truth targets.
            y(Variable): A variable of `T*B` containing ground-truth target
                token indices for the given batch.
        """

        loss = 0.0
        # Convert token indices to embeddings -> T*B*E
        y_emb = self.emb(y)

        # Get initial hidden state
        h = self.f_init(*ctx_dict['txt'])

        # -1: So that we skip the timestep where input is <eos>
        for t in range(y_emb.shape[0] - 1):
            log_p, h = self.f_next(ctx_dict, y_emb[t], h)
            loss += torch.gather(
                log_p, dim=1, index=y[t + 1].unsqueeze(1)).sum()

        return loss
