from pathlib import Path

import torch

from torch.nn.utils.rnn import pad_sequence

from ..utils.data import read_sentences
from .base import BaseDataset


class TextDataset(BaseDataset):
    r"""A PyTorch dataset for sentences.

    Arguments:
        fname (str or Path): A string or ``pathlib.Path`` object giving
            the corpus.
        vocab (Vocabulary): A ``Vocabulary`` instance for the given corpus.
        bos (bool, optional): If ``True``, a special beginning-of-sentence
            "<bos>" marker will be prepended to sentences.
        eos (bool, optional): If ``False``, end-of-sentence marker <eos>
            will not be appended to sentences.

    """
    def __init__(self, fname, vocab, bos=False, eos=True):
        # Public fields will be dumped from __repr__()
        self.path = Path(fname)
        self.vocab = vocab
        self.bos = bos
        self.eos = eos

        # Read the dataset into memory
        self._data, self._lengths, self._keys = self._read()

        # Dataset size
        self._size = len(self._data)

    def _read(self):
        """Reads the sentences and map them to vocabulary."""
        data, lengths = read_sentences(
            self.path, self.vocab, bos=self.bos, eos=self.eos)

        # map to torch tensors
        data = [torch.LongTensor(t) for t in data]

        # keys for indirect mapping are identity for plain text datasets
        keys = None

        return data, lengths, keys

    @property
    def lengths(self):
        """Returns the lengths of each element in the dataset."""
        return self._lengths

    def __getitem__(self, idx):
        """Returns the `idx`'th item where `idx` is a single integer."""
        return self._data[idx]

    def collate(self, elems):
        """Collates a batch into tensor."""
        return pad_sequence(elems)

    def __len__(self):
        """Returns the number of items in the dataset."""
        return self._size
