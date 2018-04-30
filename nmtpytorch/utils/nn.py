# -*- coding: utf-8 -*-
from torch.nn import Module


def get_rnn_hidden_state(h):
    """Returns h_t transparently regardless of RNN type."""
    return h if not isinstance(h, tuple) else h[0]


# Taken from:
# https://github.com/pytorch/pytorch/pull/5297/files
class ModuleDict(Module):
    r"""Holds submodules in a dict.
    ModuleDict can be indexed like a regular Python dict, but modules it contains
    as values are properly registered, and will be visible by all Module methods.
    Arguments:
        modules (dict, optional): a dict of keys : modules to add
    Example::
        class TwoHeadedNet(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.shared_backbone = nn.Linear(10, 10)
                self.heads = nn.ModuleDict({
                    'task1' : nn.Linear(10, 5),
                    'task2' : nn.Linear(10, 10),
                })
            def forward(self, x, task_name):
                # takes an extra `task_name` argument to determine
                # which head of the network to use
                assert task_name in ['task1', 'task2']
                x = self.shared_backbone(x)
                x = self.heads[task_name](x)
                return x
    """

    def __init__(self, modules=None):
        super(ModuleDict, self).__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key):
        return self._modules[key]

    def __setitem__(self, key, module):
        return setattr(self, key, module)

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def keys(self):
        return self._modules.keys()

    def items(self):
        return self._modules.items()

    def values(self):
        return self._modules.values()

    def get(self, key, default=None):
        return self._modules.get(key, default)

    def update(self, modules):
        r"""Updates modules from a Python dict.
        Arguments:
            modules (dict): dict of modules to append
        """
        if not isinstance(modules, dict):
            raise TypeError("ModuleDict.update should be called with a "
                            "dict, but got " + type(modules).__name__)
        for key, module in modules.items():
            self.add_module(key, module)
        return self
