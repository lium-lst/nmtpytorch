# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

from .. import FF
from . import ConditionalDecoder


class SimpleGRUDecoder(ConditionalDecoder):
    """A simple GRU decoder with a single decoder layer. It has the same
    set of parameters as the parent class except `rnn_type`."""
    def __init__(self, **kwargs):
        # Set rnn_type to GRU
        kwargs['rnn_type'] = 'gru'
        super().__init__(**kwargs)

        # Remove second GRU
        # Remove and replace hid2out since we now concatenate the
        # attention output and the hidden state
        del self.dec1, self.hid2out
        self.hid2out = FF(2 * self.hidden_size,
                          self.input_size, bias_zero=True, activ='tanh')

    def f_next(self, ctx_dict, y, h):
        """Applies one timestep of recurrence."""
        # Get hidden states from the first decoder (purely cond. on LM)
        h1 = self.dec0(y, h)

        # Apply attention
        alpha_t, z_t = self.att(h1.unsqueeze(0), *ctx_dict[self.ctx_name])

        if not self.training:
            self.history['alpha_txt'].append(alpha_t)

        # Concatenate attented source and hidden state & project
        o = self.hid2out(torch.cat((h1, z_t), dim=-1))

        # Apply dropout if any
        logit = self.do_out(o) if self.dropout_out > 0 else o

        # Transform logit to T*B*V (V: vocab_size)
        # Compute log_softmax over token dim
        log_p = F.log_softmax(self.out2prob(logit), dim=-1)

        # Return log probs and new hidden states
        return log_p, h1
