import torch

from .. import LayerNorm


class ResidualLayerNorm(torch.nn.Module):
    """Residually connected Layer Normalization layer.

    Arguments:

    Input:

    Output:
    """

    def __init__(self, model_dim, affine=True, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.affine = affine
        self.dropout = dropout

        self.norm = LayerNorm(self.model_dim, elementwise_affine=self.affine)
        self.dropout_layer = torch.nn.Dropout(self.dropout)

    def forward(self, inputs):
        # Unpack into `x` and `Sublayer(x)`
        x, f_x, mask = inputs
        return (self.norm(x + self.dropout_layer(f_x)), mask)
