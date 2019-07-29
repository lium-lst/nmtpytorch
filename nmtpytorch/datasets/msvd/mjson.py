# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path

import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import ipdb

logger = logging.getLogger('nmtpytorch')


class MSVDJSONDataset(Dataset):
    r"""A PyTorch dataset for MSVD JSON files.

    Arguments:
        fname (str or Path): A string or ``pathlib.Path`` object giving
            the corpus.
        vocab (Vocabulary): A ``Vocabulary`` instance for the given corpus.
        bos (bool, optional): If ``True``, a special beginning-of-sentence
            "<bos>" marker will be prepended to sentences.
    """

    def __init__(self, fname, vocab, bos=False, eos=False, **kwargs):
        self.path = Path(fname)
        self.vocab = vocab
        self.bos = bos

        # Detect glob patterns
        self.fnames = sorted(self.path.parent.glob(self.path.name))

        if len(self.fnames) == 0:
            raise RuntimeError('{} does not exist.'.format(self.path))
        elif len(self.fnames) > 1:
            logger.info('Multiple files found, using first: {}'.format(self.fnames[0]))

        with open(self.fnames[0]) as jf:
            data = json.load(jf)
            # Do not use images for now
            self.data,  _ = data['annotations'], data['images']

        # Number of captions = dataset size
        self.size = len(self.data)
        self._map = {}

        # Split into words
        for i in range(self.size):
            # Map tokens to integers
            self.data[i]['seq'] = self.vocab.sent_to_idxs(
                self.data[i]['caption'], explicit_bos=bos,
                explicit_eos=eos)
            self._map[i] = self.data[i]['image_id']

        self.lengths = [len(d['seq']) for d in self.data]

    @staticmethod
    def to_torch(batch):
        return pad_sequence(
            [torch.tensor(b, dtype=torch.long) for b in batch], batch_first=False)

    def __getitem__(self, idx):
        return self.data[idx]['seq']

    def __len__(self):
        return self.size

    def __repr__(self):
        s = "{} '{}' ({} sentences)\n".format(
            self.__class__.__name__, self.fnames[0].name, self.__len__())
        return s
