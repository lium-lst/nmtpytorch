# -*- coding: utf-8 -*-
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from ..utils.data import sort_batch


class TextEncoder(nn.Module):
    """A recurrent encoder with embedding layer."""
    def __init__(self, input_size, hidden_size, n_vocab, rnn_type,
                 num_layers=1, bidirectional=True,
                 dropout_rnn=0, dropout_emb=0, dropout_ctx=0):
        super().__init__()

        assert self.rnn_type in ('LSTM', 'GRU'), \
            "rnn_type should be 'lstm' or 'gru'."

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_size = self.hidden_size * 2
        self.n_vocab = n_vocab
        self.rnn_type = rnn_type.upper()
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # For dropout btw layers, only effective if num_layers > 1
        self.dropout_rnn = dropout_rnn

        # Our other custom dropouts after embeddings and annotations
        self.dropout_emb = dropout_emb
        self.dropout_ctx = dropout_ctx

        if self.dropout_emb > 0:
            self.do_emb = nn.Dropout(self.dropout_emb)
        if self.dropout_ctx > 0:
            self.do_ctx = nn.Dropout(self.dropout_ctx)

        # Create embedding layer
        self.emb = nn.Embedding(self.n_vocab, self.input_size, padding_idx=0)

        # Create encoder according to the requested type
        RNN = getattr(nn, self.rnn_type)
        self.enc = RNN(self.input_size, self.hidden_size,
                       self.num_layers, bias=True, batch_first=False,
                       dropout=self.dropout_rnn,
                       bidirectional=self.bidirectional)

    def forward(self, x):
        """Receives a Variable of indices (n_timesteps, n_samples) and
        returns their recurrent representations."""
        # sort the batch by decreasing length of sequences
        # oidxs: to recover original order
        # sidxs: idxs to sort the batch
        # slens: lengths in sorted order for pack_padded_sequence()
        oidxs, sidxs, slens, mask = sort_batch(x)

        # Fetch embeddings for the sorted batch
        embs = self.emb(x[:, sidxs])

        if self.dropout_emb > 0:
            embs = self.do_emb(embs)

        # Pack and encode
        packed_emb = pack_padded_sequence(embs, slens)

        # last_state is a tuple of (h_t, c_t) for LSTM so this does indeed
        # work transparently for both GRUs and LSTMs.
        packed_hs, last_state = self.enc(packed_emb)

        # Get hidden states and revert the order
        hs = pad_packed_sequence(packed_hs)[0][:, oidxs]

        if self.dropout_ctx > 0:
            hs = self.do_ctx(hs)

        return hs, mask
