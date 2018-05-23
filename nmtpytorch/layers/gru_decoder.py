# -*- coding: utf-8 -*-
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

from ..utils.misc import get_local_args
from . import FF, Attentionv2, RNNInitializer


class GRUDecoder(nn.Module):
    """Stacked GRU decoder using CUDNN-optimized layers."""
    def __init__(self, input_size, hidden_size, ctx_size_dict, ctx_name, n_vocab,
                 n_layers_preatt=1, n_layers_postatt=1,
                 tied_emb=False, dec_init='zero', dec_init_size=None,
                 dec_init_activ='tanh', att_type='mlp', att_activ='tanh',
                 att_bottleneck='ctx', att_temp=1.0,
                 att_mlp_bias=False, dropout_dec=0, dropout_out=0,
                 emb_maxnorm=None, emb_gradscale=False):
        super().__init__()

        # Store all the passed arguments automatically
        self.__dict__.update(get_local_args(locals()))

        # Create target embeddings
        self.emb = nn.Embedding(
            self.n_vocab, self.input_size, padding_idx=0,
            max_norm=self.emb_maxnorm, scale_grad_by_freq=self.emb_gradscale)

        if self.dec_init_size is None:
            # If no auxiliary feature will be given, the input to decoder is
            # the source annotations
            self.dec_init_size = self.ctx_size_dict[self.ctx_name]

        # Create decoder initializer
        self.decinit = RNNInitializer(
            'GRU', self.dec_init_size, self.hidden_size, self.n_layers_preatt,
            self.ctx_name, self.dec_init, self.dec_init_activ)

        # Create first decoder block that'll be used for attention
        self.rnn_pre = nn.GRU(
            self.input_size, self.hidden_size, num_layers=self.n_layers_preatt,
            dropout=self.dropout_dec)

        # Create attention layer
        self.att = Attentionv2(
            ctx_dim=self.ctx_size_dict[self.ctx_name],
            hid_dim=self.hidden_size,
            mid_dim=self.att_bottleneck,
            method=self.att_type,
            concat_outputs=True,
            mlp_bias=self.att_mlp_bias,
            mlp_activ=self.att_activ,
            temp=self.att_temp)

        if self.n_layers_postatt > 0:
            # Create second decoder block that'll process attention outputs
            # Input is the concatenation of ctx and prev hidden
            self.rnn_post = nn.GRU(
                self.hidden_size + self.ctx_size_dict[self.ctx_name],
                self.hidden_size, num_layers=self.n_layers_postatt,
                dropout=self.dropout_dec)
            hid2out_input = self.hidden_size
        else:
            hid2out_input = self.hidden_size + self.ctx_size_dict[self.ctx_name]

        # Output bottleneck: maps hidden states to target emb dim
        # This is necessary for tied embeddings support
        hid2out = FF(
            hid2out_input, self.input_size, bias_zero=True, activ='tanh')

        # Final softmax
        out2prob = FF(self.input_size, self.n_vocab)

        # Tie input embedding matrix and output embedding matrix
        if self.tied_emb:
            out2prob.weight = self.emb.weight

        # Put everything inside a Sequential()
        self.classifier = nn.Sequential(OrderedDict([
            ('hid2out', hid2out),
            ('dropout', nn.Dropout(p=self.dropout_out)),
            ('out2prob', out2prob),
        ]))

        # Create loss object
        self.nll_loss = nn.NLLLoss(size_average=False, ignore_index=0)

    def f_init(self, ctx_dict):
        """Returns the initial h_0 for the decoder."""
        self.alphas = []
        return self.decinit(ctx_dict)

    def f_next(self, ctx_dict, y, h):
        # TODO
        pass

    def forward(self, ctx_dict, y):
        """Computes the softmax outputs given source encondings and
        ground-truth target token indices. Only called during training.

        Arguments:
            ctx_dict(dict): Dictionary containing the source encodings as
                ``modality_name:Variable`. What matters for the decoder is
                the self.ctx_name key that contains a variable of `S*B*ctx_dim`
                representing the source encodings.
            y(Variable): A variable of `T*B` containing ground-truth target
                token indices for the given batch.
        """

        # Convert token indices to embeddings -> T*B*E
        y_emb = self.emb(y)

        # Get initial hidden state -> n_layers*B*H
        h_0 = self.f_init(ctx_dict)

        # Run pre-attention RNN to get the hidden states
        # Don't feed the last timestep
        out1, _ = self.rnn_pre(y_emb[:-1], h_0)

        # Get inputs for the next RNN block
        scores, contexts = self.att(out1, *ctx_dict[self.ctx_name])

        if self.n_layers_postatt > 0:
            # We don't initialize this second RNN block
            out2, _ = self.rnn_post(contexts)
        else:
            out2 = contexts

        # Compute log_softmax over token dim
        logps = F.log_softmax(self.classifier(out2), dim=-1)

        loss = self.nll_loss(logps.view(-1, logps.size(2)), y[1:].view(-1))
#         loss = sum(
            # [self.nll_loss(logps[t], y[t + 1]) for t in range(y.shape[0] - 1)])

        return {
            'loss': loss,
            'logps': None if self.training else logps.data,
        }
