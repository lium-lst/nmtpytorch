# -*- coding: utf-8 -*-
import torch.nn.functional as F

from ...utils.nn import get_rnn_hidden_state
from ..attention import HierarchicalAttention, get_attention
from .. import Fusion
from . import ConditionalDecoder


class ConditionalMMDecoder(ConditionalDecoder):
    """A conditional multimodal decoder with multimodal attention."""
    def __init__(self, fusion_type='concat', aux_ctx_name='image', **kwargs):
        super().__init__(**kwargs)
        self.aux_ctx_name = aux_ctx_name

        # Define (context) fusion operator
        self.fusion_type = fusion_type
        if fusion_type == "hierarchical":
            self.fusion = HierarchicalAttention(
                [self.hidden_size, self.hidden_size],
                self.hidden_size, self.hidden_size)
        else:
            self.fusion = Fusion(
                fusion_type, 2 * self.hidden_size, self.hidden_size)

        # Rename textual attention layer
        self.txt_att = self.att
        del self.att

        # Visual attention over convolutional feature maps
        Attention = get_attention(self.att_type)
        self.img_att = Attention(
            self.ctx_size_dict[self.aux_ctx_name], self.hidden_size,
            transform_ctx=self.transform_ctx, mlp_bias=self.mlp_bias,
            att_activ=self.att_activ,
            att_bottleneck=self.att_bottleneck)

    def f_next(self, ctx_dict, y, h):
        # Get hidden states from the first decoder (purely cond. on LM)
        h1_c1 = self.dec0(y, self._rnn_unpack_states(h))
        h1 = get_rnn_hidden_state(h1_c1)

        # Apply attention
        self.txt_alpha_t, txt_z_t = self.txt_att(
            h1.unsqueeze(0), *ctx_dict[self.ctx_name])
        self.img_alpha_t, img_z_t = self.img_att(
            h1.unsqueeze(0), *ctx_dict[self.aux_ctx_name])
        # Save for reg loss terms
        self.history['alpha_img'].append(self.img_alpha_t.unsqueeze(0))

        # Context will double dimensionality if fusion_type is concat
        # z_t should be compatible with hidden_size
        if self.fusion_type == "hierarchical":
            self.h_att, z_t = self.fusion([txt_z_t, img_z_t], h1.unsqueeze(0))
        else:
            z_t = self.fusion(txt_z_t, img_z_t)

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
