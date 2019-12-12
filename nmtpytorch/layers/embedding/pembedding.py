# -*- coding: utf-8 -*-
from torch import nn

from .. import FF


class PEmbedding(nn.Embedding):
    """An extension to regular `nn.Embedding` with MLP and dropout."""
    def __init__(self, num_embeddings, embedding_dim, out_dim,
                 activ='linear', dropout=0.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx=0)
        self.proj = FF(embedding_dim, out_dim, activ=activ, bias=False)
        self.do = nn.Dropout(dropout) if dropout > 0.0 else lambda x: x

    def forward(self, input):
        # Get the embeddings from parent's forward
        return self.do(self.proj(super().forward(input)))
