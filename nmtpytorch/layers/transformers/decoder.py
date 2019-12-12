import torch

from ..attention import ScaledDotAttention
from . import ResidualLayerNorm, PositionwiseFF


class TFDecoder(torch.nn.Module):
    """Decoder block for Transformer.

    Arguments:

    Input:

    Output:
    """

    def __init__(self, model_dim, ff_dim, n_heads, n_layers):
        super().__init__()
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        blocks = []

        for _ in range(self.n_layers):
            layers = torch.nn.Sequential(
                ScaledDotAttention(self.model_dim, self.n_heads, causal=True),
                ResidualLayerNorm(self.model_dim),
                PositionwiseFF(self.model_dim, self.ff_dim),
                ResidualLayerNorm(self.model_dim),
            )
            blocks.append(layers)

        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, x, mask=None, **kwargs):
        """Forward-pass of the encoder block.

        :param x: input tensor, shape (tstep, bsize, model_dim)
        :param mask: mask tensor for unavailable batch positions (tstep, bsize)

        :return: foo
        """
        for block in self.blocks:
            x, mask = block((x, x, x, mask))
        return (x, mask)
