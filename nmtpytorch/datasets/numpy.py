# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    r"""A PyTorch dataset for Numpy .npy/npz serialized tensor files. The
    serialized tensor's first dimension should be the batch dimension.

    Arguments:
        fname (str or Path): A string or ``pathlib.Path`` object for
            the relevant numpy file.
        key (str, optional): If `fname` is `.npz` file, its relevant `key`
            will be fetched from the serialized object.
    """

    def __init__(self, fname, key=None):
        self.path = Path(fname)
        if not self.path.exists():
            raise RuntimeError('{} does not exist.'.format(self.path))

        if self.path.suffix == '.npy':
            self.data = np.load(self.path)
        elif self.path.suffix == '.npz':
            assert key, "A key should be provided for .npz files."
            self.data = np.load(self.path)[key]

        # Dataset size
        self.size = self.data.shape[0]

    @staticmethod
    def to_torch(batch):
        return torch.from_numpy(np.array(batch, dtype='float32'))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.size

    def __repr__(self):
        s = "{} '{}' ({} samples)\n".format(
            self.__class__.__name__, self.path.name, self.__len__())
        return s
