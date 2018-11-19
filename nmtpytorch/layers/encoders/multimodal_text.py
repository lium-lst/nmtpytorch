# -*- coding: utf-8 -*-
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from ...utils.data import sort_batch

from . import TextEncoder


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
        self.n_init_types = 2 if self.rnn_type == 'LSTM' else 1

        ##################################################
        # Create the necessary visual transformation layer
        ##################################################
        if self.feat_fusion == 'init':
            # Use single h_0/c_0 for all stacked layers and directions for a
            # consistent information source.
            self.ff_vis = FF(
                self.feat_size,
                self.hidden_size * self.n_init_types, activ=self.feat_activ)
        elif self.feat_fusion in ('concat', 'sum', 'prepend', 'append', 'preappend'):
            # These does not differentiate between the transformation layer
            self.ff_vis = FF(
                self.feat_size, self.input_size,
                activ=self.feat_activ)
        elif self.feat_fusion == 'concat_fuse':
            # Concatenates and fuses back to same input dimension
            # to avoid doubling RNN input dimensionality
            self.ff_vis = FF(
                self.feat_size + self.input_size, self.input_size,
                activ=self.feat_activ)
        ###############################################
        # Set integration func to avoid if-else clutter
        ###############################################
        if self.feat_fusion == 'sum':
            #self.merge_op = lambda e, v:
            pass

    def forward(self, x, v, **kwargs):
        # Transform the visual vector
        v = self.ff_vis(v)

        # Let's do pack/pad all the time to cut down boilerplate code
        oidxs, sidxs, slens, mask = sort_batch(x)

        # Fetch embeddings for the sorted batch
        embs = self.emb(x[:, sidxs])

        if self.dropout_emb > 0:
            embs = self.do_emb(embs)

        # Pack and encode
        packed_emb = pack_padded_sequence(embs, slens)

        # We ignore last_state since we don't use it
        packed_hs, _ = self.enc(packed_emb, h0)

        # Get hidden states and revert the order
        hs = pad_packed_sequence(packed_hs)[0][:, oidxs]

        return self.output(hs), mask
