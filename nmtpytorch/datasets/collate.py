# -*- coding: utf-8 -*-
from collections import UserDict

import numpy as np
import torch

from ..utils.data import pad_data, onehot_data


def get_collate(data_sources):
    """Returns a special collate_fn which will view the underlying data
    in terms of the given DataSource keys."""

    def collate_fn(batch):
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
                # Underlying data is already converted Torch tensor
                tensors[ds] = torch.stack(batch_data)
            elif ds._type == "Numpy":
                tensors[ds] = torch.from_numpy(
                    np.array(batch_data, dtype='float32'))

        return tensors
    return collate_fn
