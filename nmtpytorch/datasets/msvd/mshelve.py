# -*- coding: utf-8 -*-
import shelve
import logging
from pathlib import Path

import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import ipdb

logger = logging.getLogger('nmtpytorch')


class MSVDShelveDataset(Dataset):
    r"""A PyTorch dataset for MSVD Shelve files.

    Arguments:
        fname (str or Path): A string or ``pathlib.Path`` object giving
            the corpus.
        mode (str): Provided through **kwargs. Determines how the features
            should be constructed.
    """

    def __init__(self, fname, **kwargs):
        self.path = Path(fname)
        if not self.path.with_suffix(self.path.suffix + '.dir').exists():
            raise RuntimeError('{}* does not exist.'.format(self.path))

        # Underlying data object
        self.data = shelve.open(str(self.path.resolve()), flag='r')
        self.size = len(self.data)

        # This will be modified from outside if a coordinating master dataset exists
        self.idx2key = lambda x: x

#         self.labels = []
        # for v in self.data.values():
            # for feats in list(v.values()):
                # for feat in feats:
                    # self.labels.append(feat[0])

        # Fetch # of category labels
        # ipdb.set_trace()

    @staticmethod
    def to_torch(batch):
        return pad_sequence(
            [torch.tensor(b, dtype=torch.long) for b in batch], batch_first=False)

    def __getitem__(self, idx):
        feats = self.data[self.idx2key(idx)]
        # The above gives a {'frame_cnt.jpg' : list of detected objs}
        feats = [feats[sidx][0][0] if len(feats[sidx]) > 0 else 0 for sidx in sorted(feats.keys())]

        # This will always return a 16*batch_size category IDs for each 16th frame
        return feats

    def __len__(self):
        return self.size

    def __repr__(self):
        s = "{} '{}' ({} samples)\n".format(
            self.__class__.__name__, self.path, self.__len__())
        return s
