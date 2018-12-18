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
        order_file (str, None): If given, will be used to map sample indices
            to tensors using this list. Useful for tiled or repeated
            experiments.
        revert (bool, optional): If `True`, the data order will be reverted
            for adversarial/incongruent experiments during test-time.
    """

    def __init__(self, fname, key=None, order_file=None, revert=False, **kwargs):
        self.path = Path(fname)
        if not self.path.exists():
            raise RuntimeError('{} does not exist.'.format(self.path))

        if self.path.suffix == '.npy':
            self.data = np.load(self.path)
        elif self.path.suffix == '.npz':
            assert key, "A key should be provided for .npz files."
            self.data = np.load(self.path)[key]

        if order_file:
            with open(order_file) as orf:
                self.order = [int(x) for x in orf.read().strip().split('\n')]
        else:
            self.order = list(range(self.data.shape[0]))

        if revert:
            self.order = self.order[::-1]

        # Dataset size
        self.size = len(self.order)

    @staticmethod
    def to_torch(batch):
        # NOTE: Assumes x.shape == (n, *)
        x = torch.from_numpy(np.array(batch, dtype='float32'))
        # Convert it to (t(=1 if fixed features), n, c)
        # By default we flatten h*w to first dim for interoperability
        # Models should further reshape the tensor for their needs
        return x.view(*x.size()[:2], -1).permute(2, 0, 1)

    def __getitem__(self, idx):
        return self.data[self.order[idx]]

    def __len__(self):
        return self.size

    def __repr__(self):
        s = "{} '{}' ({} samples)\n".format(
            self.__class__.__name__, self.path.name, self.__len__())
        return s
