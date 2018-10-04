# -*- coding: utf-8 -*-
from ..utils.device import DEVICE


class Batch(dict):
    """A custom dictionary that does on-the-fly device conversion."""
    def __init__(self, data_dict, size):
        self.size = size
        self.update({str(k): v.to(DEVICE) for k, v in data_dict.items()})

    def __repr__(self):
        s = "Batch(size={})\n".format(self.size)
        for data_source, tensor in self.items():
            s += "  {:10s} -> {} - {}\n".format(
                data_source, tensor.shape, tensor.device)
        return s


def get_collate(data_sources):
    """Returns a special collate_fn which will view the underlying data
    in terms of the given DataSource keys."""

    def collate_fn(batch):
        return Batch(
            {ds: ds.to_torch([elem[ds] for elem in batch]) for ds in data_sources},
            len(batch),
        )

    return collate_fn
