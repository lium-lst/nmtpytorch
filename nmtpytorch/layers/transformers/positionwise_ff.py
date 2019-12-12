import torch

from .. import FF


class PositionwiseFF(torch.nn.Module):
    """Positionwise Feed-forward layer.

    Arguments:

    Input:

    Output:
    """

    def __init__(self, model_dim, ff_dim, activ='relu'):
        super().__init__()
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.activ = activ

        # Create the layers
        self.func = torch.nn.Sequential(
            FF(self.model_dim, self.ff_dim, activ=self.activ),
            FF(self.ff_dim, self.model_dim, activ=None),
        )

    def forward(self, inputs):
        x, mask = inputs
        return (x, self.func(x), mask)
