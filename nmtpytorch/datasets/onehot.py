# -*- coding: utf-8 -*-
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..utils.data import read_sentences, onehot_data


class OneHotDataset(Dataset):
    r"""A PyTorch dataset for one-hot encoded binary attributes. The actual
        encoding will be done in the relevant collate function.

    Arguments:
        fname (str or Path): A string or ``pathlib.Path`` object giving
            space delimited attributes per sentence.
        vocab (Vocabulary): A ``Vocabulary`` instance for the attributes.
    """

    def __init__(self, fname, vocab, **kwargs):
        self.path = Path(fname)
        self.vocab = vocab

        # Detect glob patterns
        self.fnames = sorted(self.path.parent.glob(self.path.name))

        if len(self.fnames) == 0:
            raise RuntimeError('{} does not exist.'.format(self.path))
        elif len(self.fnames) > 1:
            raise RuntimeError("Multiple source files not supported.")

        # Read the sentences and map them to vocabulary
        self.data, self.lengths = read_sentences(
            self.fnames[0], self.vocab, eos=False, bos=False)

        # Convert indices to torch tensors
        self.data = [torch.LongTensor(elem) for elem in self.data]

        # number of possible classes is the vocab size
        self.n_classes = len(self.vocab)

        # Dataset size
        self.size = len(self.data)

    @staticmethod
    def to_torch(batch, **kwargs):
        return onehot_data(batch, **kwargs)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.size

    def __repr__(self):
        s = "{} '{}' ({} sentences)\n".format(
            self.__class__.__name__, self.fnames[0].name, self.__len__())
        return s
