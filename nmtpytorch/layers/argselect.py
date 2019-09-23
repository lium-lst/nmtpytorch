import torch


class ArgSelect(torch.nn.Module):
    """Dummy layer that picks one of the returned values from mostly RNN-type
    `nn.Module` layers."""
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]

    def __repr__(self):
        return "ArgSelect(index={})".format(self.index)
