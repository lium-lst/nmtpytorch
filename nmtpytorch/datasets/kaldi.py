# -*- coding: utf-8 -*-
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from ..utils.kaldi import readMatrixShape, readMatrixByOffset

# TODO
# ----
# an lru_cache() decorated version of readMatrixByOffset() will make sure that
# all the training data is cached into memory after 1 epoch.


class KaldiDataset(Dataset):
    """A PyTorch dataset for Kaldi .scp/ark.

    Arguments:
        fname (str or Path): A string or ``pathlib.Path`` object for
            a folder that contains ``feats_local.scp`` and optionally a ``segments.len``
            file containing segment lengths.
    """

    def __init__(self, fname, **kwargs):
        self.data = []
        self.lengths = []
        self.root = Path(fname)
        self.scp_path = self.root / 'feats_local.scp'
        self.len_path = self.root / 'segments.len'

        if not self.scp_path.exists():
            raise RuntimeError('{} does not exist.'.format(self.scp_path))

        if self.len_path.exists():
            read_lengths = False
            # Read lengths file
            with open(self.len_path) as f:
                for line in f:
                    self.lengths.append(int(line.strip()))
        else:
            # Read them below (this is slow)
            read_lengths = True

        with open(self.scp_path) as scp_input_file:
            for line in tqdm(scp_input_file, unit='segments'):
                uttid, pointer = line.strip().split()
                arkfile, offset = pointer.rsplit(':', 1)
                offset = int(offset)
                self.data.append((arkfile, offset))
                if read_lengths:
                    with open(arkfile, "rb") as g:
                        g.seek(offset)
                        feat_len = readMatrixShape(g)[0]

                    self.lengths.append(feat_len)

        # Set dataset size
        self.size = len(self.data)

        if self.size != len(self.lengths):
            raise RuntimeError("Dataset size and lengths size does not match.")

    @staticmethod
    def to_torch(batch):
        return pad_sequence(
            [torch.FloatTensor(x) for x in batch], batch_first=False)

    def __getitem__(self, idx):
        """Read segment features from the actual .ark file."""
        return readMatrixByOffset(*self.data[idx])

    def __len__(self):
        return self.size

    def __repr__(self):
        s = "{} '{}' ({} samples)\n".format(
            self.__class__.__name__, self.scp_path.name, self.__len__())
        return s
