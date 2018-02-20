# -*- coding: utf-8 -*-
from collections import UserDict

import numpy as np
import torch

from ..utils.data import pad_data, onehot_data


def get_collate_v1(keys):
    """Returns a special collate_fn which will view the underlying data
    in terms of the given keys."""

    def collate_fn(batch):
        tensors = UserDict()
        tensors.size = len(batch)

        # Iterate over data sources
        for key in keys:
            if isinstance(batch[0][key], list):
                # Sequence vocabulary indices
                tensors[key] = pad_data([elem[key] for elem in batch])
            elif isinstance(batch[0][key], np.ndarray):
                # Image features data from .npy
                tensors[key] = torch.stack([torch.from_numpy(elem[key])
                                            for elem in batch])

        return tensors
    return collate_fn


def get_collate_v2(data_sources):
    """Returns a special collate_fn which will view the underlying data
    in terms of the given DataSource keys."""

    def collate_fn_v2(batch):
        tensors = UserDict()
        tensors.size = len(batch)

        # Iterate over keys which are DataSource objects
        for ds in data_sources:
            batch_data = [elem[ds] for elem in batch]
            if ds._type == "Text":
                tensors[ds] = pad_data(batch_data)
            elif ds._type == "OneHot":
                # Hack: we inject n_classes into DataSource keys
                # from the model itself.
                tensors[ds] = onehot_data(batch_data, ds._n_classes)
            elif ds._type == "ImageFolder":
                tensors[ds] = torch.stack(batch_data)

        return tensors
    return collate_fn_v2
