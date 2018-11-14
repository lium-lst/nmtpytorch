# -*- coding: utf-8 -*-
import torch.nn.functional as F

from ...utils.nn import get_rnn_hidden_state
from . import ConditionalDecoder

# Decoder without attention that uses a single input vector.
# Layer contributed by @loicbarrault


class VectorDecoder(ConditionalDecoder):
    """Single-layer RNN decoder using fixed-size vector representation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Remove attention layer and the second decoder
        del self.att, self.dec1

    def f_next(self, ctx_dict, y, h):
        """Applies one timestep of recurrence."""
        # Get hidden states from the decoder
        h1_c1 = self.dec0(y, self._rnn_unpack_states(h))
        h1 = get_rnn_hidden_state(h1_c1)

        # Project hidden state to embedding size
        o = self.hid2out(h1)

        # Apply dropout if any
        logit = self.do_out(o) if self.dropout_out > 0 else o

        # Transform logit to T*B*V (V: vocab_size)
        # Compute log_softmax over token dim
        log_p = F.log_softmax(self.out2prob(logit), dim=-1)

        # Return log probs and new hidden states
        return log_p, self._rnn_pack_states(h1_c1)
