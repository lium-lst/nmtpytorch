# -*- coding: utf-8 -*-
from torch import nn
import torch.nn.functional as F

from ...utils.nn import get_rnn_hidden_state
from ..attention import get_attention, HierarchicalAttention
from .. import Fusion
from . import ConditionalDecoder


class MultiSourceConditionalDecoder(ConditionalDecoder):
    """A conditional multimodal decoder with multimodal attention."""
    def __init__(self, ctx_names, fusion_type='concat', **kwargs):
        super().__init__(**kwargs)

        # Define (context) fusion operator
        self.ctx_names = ctx_names
        self.fusion_type = fusion_type
        if fusion_type == "hierarchical":
            self.fusion = HierarchicalAttention(
                [self.hidden_size for _ in ctx_names],
                self.hidden_size, self.hidden_size)
        else:
            raise NotImplementedError("Concatenation and sum work only with two inputs now.")
            self.fusion = Fusion(
                fusion_type, len(ctx_names) * self.hidden_size, self.hidden_size)

        attns = []
        for ctx_name in ctx_names:
            Attention = get_attention(self.att_type)
            attns.append(Attention(
                self.ctx_size_dict[ctx_name], self.hidden_size,
                transform_ctx=self.transform_ctx, mlp_bias=self.mlp_bias,
                att_activ=self.att_activ,
                att_bottleneck=self.att_bottleneck))
        self.attns = nn.ModuleList(attns)

    def f_next(self, ctx_dict, y, h):
        # Get hidden states from the first decoder (purely cond. on LM)
        h1_c1 = self.dec0(y, self._rnn_unpack_states(h))
        h1 = get_rnn_hidden_state(h1_c1)

        # Apply attention
        ctx_list = [att(h1.unsqueeze(0), *ctx_dict[name])[1]
                    for att, name in zip(self.attns, self.ctx_names)]

        # Context will double dimensionality if fusion_type is concat
        # z_t should be compatible with hidden_size
        if self.fusion_type == "hierarchical":
            _, z_t = self.fusion(ctx_list, h1.unsqueeze(0))
        else:
            z_t = self.fusion(ctx_list)

        # Run second decoder (h1 is compatible now as it was returned by GRU)
        h2_c2 = self.dec1(z_t, h1_c1)
        h2 = get_rnn_hidden_state(h2_c2)

        # This is a bottleneck to avoid going from H to V directly
        logit = self.hid2out(h2)

        # Apply dropout if any
        if self.dropout_out > 0:
            logit = self.do_out(logit)

        # Transform logit to T*B*V (V: vocab_size)
        # Compute log_softmax over token dim
        log_p = F.log_softmax(self.out2prob(logit), dim=-1)

        # Return log probs and new hidden states
        return log_p, self._rnn_pack_states(h2_c2)
