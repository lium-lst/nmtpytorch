# -*- coding: utf-8 -*-
import torch

from . import TextEncoder
from .. import FF


class MultimodalTextEncoder(TextEncoder):
    """A multimodal recurrent encoder with embedding layer.

    Arguments:
        feat_size (int): Auxiliary feature dimensionality.
        feat_fusion(str): Type of feature fusion: 'early_concat', 'early_sum',
            'late_concat', 'late_sum', 'init'.
        feat_activ(str): Type of non-linearity if any for feature projection
            layer.
        input_size (int): Embedding dimensionality.
        hidden_size (int): RNN hidden state dimensionality.
        n_vocab (int): Number of tokens for the embedding layer.
        rnn_type (str): RNN Type, i.e. GRU or LSTM.
        num_layers (int, optional): Number of stacked RNNs (Default: 1).
        bidirectional (bool, optional): If `False`, the RNN is unidirectional.
        dropout_rnn (float, optional): Inter-layer dropout rate only
            applicable if `num_layers > 1`. (Default: 0.)
        dropout_emb(float, optional): Dropout rate for embeddings (Default: 0.)
        dropout_ctx(float, optional): Dropout rate for the
            encodings/annotations (Default: 0.)
        emb_maxnorm(float, optional): If given, renormalizes embeddings so
            that their norm is the given value.
        emb_gradscale(bool, optional): If `True`, scales the gradients
            per embedding w.r.t. to its frequency in the batch.
        proj_dim(int, optional): If not `None`, add a final projection
            layer. Can be used to adapt dimensionality for decoder.
        proj_activ(str, optional): Non-linearity for projection layer.
            `None` or `linear` does not apply any non-linearity.
        layer_norm(bool, optional): Apply layer normalization at the
            output of the encoder.

    Input:
        x (Tensor): A tensor of shape (n_timesteps, n_samples)
            including the integer token indices for the given batch.
        v (Tensor): A tensor of shape (...) representing a fixed-size
            visual vector for the batch.

    Output:
        hs (Tensor): A tensor of shape (n_timesteps, n_samples, hidden)
            that contains encoder hidden states for all timesteps. If
            bidirectional, `hs` is doubled in size in the last dimension
            to contain both directional states.
        mask (Tensor): A binary mask of shape (n_timesteps, n_samples)
            that may further be used in attention and/or decoder. `None`
            is returned if batch contains only sentences with same lengths.
    """
    def __init__(self, feat_size, feat_fusion, feat_activ=None, **kwargs):
        super().__init__(**kwargs)
        self.feat_size = feat_size
        self.feat_fusion = feat_fusion
        self.feat_activ = feat_activ

        # LSTM requires the initialization of both c_0 and h_0
        # FIXME: Not tested at all with LSTMs, probably won't work!
        self.n_init_types = 2 if self.rnn_type == 'LSTM' else 1

        ##################################################
        # Create the necessary visual transformation layer
        ##################################################
        self.plain = self.feat_fusion is None
        self.init_enc = self.feat_fusion in ('encinit', 'encdecinit')
        # No-op by default
        self.merge_op = lambda e, *v: e

        if self.init_enc:
            self.tile_factor = self.num_layers
            if self.bidirectional:
                self.tile_factor *= 2
            out_dim = self.hidden_size * self.n_init_types
            inp_dim = self.feat_size
        elif self.feat_fusion in ('concat', 'sum', 'prepend', 'append'):
            out_dim = self.input_size
            inp_dim = self.feat_size
            if self.feat_fusion == 'concat':
                inp_dim += self.input_size
                self.merge_op = lambda e, v: self.ff_vis(torch.cat(
                    (e, v.expand(e.shape[0], -1, -1)), dim=-1))
            elif self.feat_fusion == 'sum':
                self.merge_op = lambda e, v: e + self.ff_vis(v)
            elif self.feat_fusion == 'prepend':
                self.merge_op = lambda e, v: torch.cat((self.ff_vis(v), e), dim=0)
            elif self.feat_fusion == 'append':
                # NOTE: note that it will append after <eos>
                self.merge_op = lambda e, v: torch.cat((e, self.ff_vis(v)), dim=0)

        if not self.plain:
            self.ff_vis = FF(inp_dim, out_dim, activ=self.feat_activ)

    def forward(self, x, v, **kwargs):
        h0 = None
        if self.init_enc:
            h0 = self.ff_vis(v).expand(self.tile_factor, -1, -1).contiguous()

        # Compute mask for possible paddings
        zero_pos = (x == 0)
        mask = (~zero_pos).long() if zero_pos.nonzero().numel() else None

        # Fetch embeddings
        embs = self.emb(x)

        # Fuse
        embs = self.merge_op(embs, v)

        if mask is not None and embs.shape[0] != mask.shape[0]:
            # prepend, append will cause this, enlarge the mask
            mask = torch.cat((mask[0].unsqueeze(0), mask), dim=0)

        # Apply dropout
        if self.dropout_emb > 0:
            embs = self.do_emb(embs)

        # Encode
        hs, _ = self.enc(embs, h0)

        # Return
        return self.output(hs), mask
