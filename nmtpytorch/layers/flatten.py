import torch


class Flatten(torch.nn.Module):
    """A flatten module to squeeze single dimensions."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

    def __repr__(self):
        return "Flatten()"
