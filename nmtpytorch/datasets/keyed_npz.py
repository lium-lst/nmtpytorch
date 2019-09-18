# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class KeyedNPZDataset(Dataset):
    r"""A PyTorch dataset for numpy .npz file which contains a key:value store.
    Keys define the sample IDs while values are single feature tensors.

    Every value should contain a `1` in the first batch dimension while
    there may be arbitrarily more dimensions afterwards. For example,
    each tensor may be 3D if for every sample the feature is a matrix or
    may be 2D if single vectors are provided as features.

    Arguments:
        fname (str or Path): A string or ``pathlib.Path`` object for the .npz file.
        **kwargs (optional): Any other arguments.
    """

    def __init__(self, fname, **kwargs):
        self.path = Path(fname)
        if not self.path.exists():
            raise RuntimeError('{} does not exist.'.format(self.path))

        # Load the file into .data
        self.data = np.load(self.path)
        self.keys = sorted(self.data.files)
        self.size = len(self.keys)

        # Introspect to determine feature size
        feat_shape = self.data[self.keys[0]].shape

        # Do not count the first dimension
        self.ndim = len(feat_shape) - 1

        # Cache the data
        self.data = {key: self.data[key] for key in self.keys}

        self.lengths = None
        if self.ndim == 2:
            # Introspect the sequence lengths
            self.lengths = [self.data[key].shape[1] for key in self.keys]

    @staticmethod
    def to_torch(batch, **kwargs):
        # NOTE: Assumes x.shape == (n, *) & make batch the 1st dim
        x = torch.from_numpy(np.array(batch, dtype='float32'))
        return x.squeeze_(1).view(
            x.shape[0],
            x.shape[1] if x.ndimension() == 3 else 1,
            x.shape[-1]).permute(1, 0, 2)

    def __getitem__(self, idx):
        """Fetch a sample from the dataset with string keys."""
        return self.data[self.keys[idx]]

    def __len__(self):
        """Returns the dataset size."""
        return self.size

    def __repr__(self):
        s = "{} '{}' ({} samples, {}-dim features)\n".format(
            self.__class__.__name__, self.path.name, self.__len__(), self.ndim)
        return s
