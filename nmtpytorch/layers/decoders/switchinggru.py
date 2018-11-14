# -*- coding: utf-8 -*-
from collections import defaultdict
import torch
from torch import nn
import torch.nn.functional as F

from .. import FF
from ...utils.device import DEVICE
from ..attention import get_attention


class SwitchingGRUDecoder(nn.Module):
    """A multi-source aware attention based decoder. During training,
        this decoder will be fed by a single modality at a time while
        during inference one of the src->trg tasks will be performed.
    """
    def __init__(self, input_size, hidden_size, modality_dict, n_vocab,
                 tied_emb=False, dropout_out=0):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.n_vocab = n_vocab
        self.tied_emb = tied_emb
        self.dropout_out = dropout_out

        # Will have N attentions for N possible input modalities
        # dict: {en_speech: (encoding_size, att_type)}
        atts = {}
        for name, (enc_size, att_type) in modality_dict.items():
            atts[name] = get_attention(att_type)(enc_size, self.hidden_size)

        self.atts = nn.ModuleDict(atts)

        # Create target embeddings
        self.emb = nn.Embedding(self.n_vocab, self.input_size, padding_idx=0)

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

        # Final loss
        self.nll_loss = nn.NLLLoss(reduction="sum", ignore_index=0)

    def f_init(self, sources):
        """Returns the initial h_0 for the decoder. `sources` is not used
        but passed for compatibility with beam search."""
        self.history = defaultdict(list)
        batch_size = next(iter(sources.values()))[0].shape[1]
        # NOTE: Non-scatter aware, fix this
        return torch.zeros(batch_size, self.hidden_size, device=DEVICE)

    def f_next(self, sources, y, h):
        # Get hidden states from the first decoder (purely cond. on LM)
        h_1 = self.dec0(y, h)

        # sources will always contain single modality
        assert len(sources) == 1
        modality = list(sources.keys())[0]

        # Apply modality-specific attention
        alpha_t, z_t = self.atts[modality](h_1.unsqueeze(0), *sources[modality])
        self.history['alpha_{}'.format(modality)].append(alpha_t)

        # Run second decoder (h_1 is compatible now as it was returned by GRU)
        h_2 = self.dec1(z_t, h_1)

        # This is a bottleneck to avoid going from H to V directly
        logit = self.hid2out(h_2)

        # Apply dropout if any
        if self.dropout_out > 0:
            logit = self.do_out(logit)

        # Transform logit to T*B*V (V: vocab_size)
        # Compute log_softmax over token dim
        log_p = F.log_softmax(self.out2prob(logit), dim=-1)

        # Return log probs and new hidden states
        return log_p, h_2

    def forward(self, sources, y):
        """Computes the softmax outputs given source annotations `sources` and
        ground-truth target token indices `y`. Only called during training.

        Arguments:
            sources(Tensor): A tensor of `S*B*ctx_dim` representing the source
                annotations in an order compatible with ground-truth targets.
            y(Tensor): A tensor of `T*B` containing ground-truth target
                token indices for the given batch.
        """

        loss = 0.0
        logps = None if self.training else torch.zeros(
            y.shape[0] - 1, y.shape[1], self.n_vocab, device=y.device)

        # Convert token indices to embeddings -> T*B*E
        y_emb = self.emb(y)

        # Get initial hidden state
        h = self.f_init(sources)

        # -1: So that we skip the timestep where input is <eos>
        for t in range(y_emb.shape[0] - 1):
            log_p, h = self.f_next(sources, y_emb[t], h)
            if not self.training:
                logps[t] = log_p.data
            loss += self.nll_loss(log_p, y[t + 1])

        return {'loss': loss, 'logps': logps}
