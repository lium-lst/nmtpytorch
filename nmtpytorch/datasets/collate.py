# -*- coding: utf-8 -*-
from collections import UserDict


def get_collate(data_sources):
    """Returns a special collate_fn which will view the underlying data
    in terms of the given DataSource keys."""

    def collate_fn(batch):
        tensors = UserDict()
        tensors.size = len(batch)

        # Iterate over keys which are DataSource objects
        for ds in data_sources:
            batch_data = [elem[ds] for elem in batch]
            tensors[ds] = ds.to_torch(batch_data)

        return tensors
    return collate_fn
