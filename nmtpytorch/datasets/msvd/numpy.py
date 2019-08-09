# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MSVDNumpyDataset(Dataset):
    r"""A PyTorch dataset for Numpy .npz files. The data tensor should be
    in the 'data' key. List of sample identifiers should be provided in the
    'keys' key.
    The serialized tensor's first dimension should be the batch dimension.

    Arguments:
        fname (str or Path): A string or ``pathlib.Path`` object for
            the relevant numpy file.
    """

    def __init__(self, fname, **kwargs):
        self.path = Path(fname)
        if not self.path.exists():
            raise RuntimeError('{} does not exist.'.format(self.path))

        # Load the file
        npz_file = np.load(self.path)
        self.data = npz_file['data']
        self._keys = npz_file['keys'].tolist()

        # sample ID to local index
        self.key2idx = {key: i for i, key in enumerate(self._keys)}
        self.idx2key = lambda x: self._keys[x]

        # Dataset size
        self.size = self.data.shape[0]

    @staticmethod
    def to_torch(batch):
        # NOTE: Assumes x.shape == (n, *) & make batch the 1st dim
        x = torch.from_numpy(np.array(batch, dtype='float32'))
        return x.view(
            x.shape[0],
            x.shape[1] if x.ndimension() == 3 else 1,
            x.shape[-1]).permute(1, 0, 2)

    def __getitem__(self, idx):
        # The sample order matches the keys given in the .npz file so
        # we can safely index using plain integers here
        return self.data[self.key2idx[self.idx2key(idx)]]

    def __len__(self):
        return self.size

    def __repr__(self):
        s = "{} '{}' ({} samples)\n".format(
            self.__class__.__name__, self.path.name, self.__len__())
        return s
