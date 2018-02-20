# -*- coding: utf-8 -*-
from pathlib import Path

from torch.utils.data import Dataset

from ..utils.data import read_sentences


class TextDataset(Dataset):
    r"""A PyTorch dataset for sentences.

    Arguments:
        fname (str or Path): A string or ``pathlib.Path`` object giving
            the corpus.
        vocab (Vocabulary): A ``Vocabulary`` instance for the given corpus.
        bos (bool, optional): If ``True``, a special beginning-of-sentence
            "<bos>" marker will be prepended to sentences.
    """

    def __init__(self, fname, vocab, bos=False):
        self.path = Path(fname)
        self.vocab = vocab
        self.bos = bos

        # Detect glob patterns
        self.fnames = sorted(self.path.parent.glob(self.path.name))

        if len(self.fnames) == 0:
            raise RuntimeError('{} does not exist.'.format(self.path))
        elif len(self.fnames) > 1:
            raise RuntimeError("Multiple source files not supported.")

        # Read the sentences and map them to vocabulary
        self.data, self.lengths = read_sentences(
            self.fnames[0], self.vocab, bos=self.bos)

        # Dataset size
        self.size = len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.size

    def __repr__(self):
        s = "{} ({} sentences)\n".format(
            self.__class__.__name__, self.__len__())
        s += " {}".format(self.fnames[0].name)
        return s
