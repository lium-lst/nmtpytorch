# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path

import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger('nmtpytorch')


class VatexJSONDataset(Dataset):
    r"""A PyTorch dataset for VaTeX .json files.

    Arguments:
        fname (str or Path): A string or ``pathlib.Path`` object giving
            the corpus.
        vocab (Vocabulary): A ``Vocabulary`` instance for the given corpus.
        bos (bool, optional): If ``True``, a special beginning-of-sentence
            "<bos>" marker will be prepended to sentences.
    """

    def __init__(self, fname, vocab, bos=True, eos=True, **kwargs):
        self.path = Path(fname)
        self.vocab = vocab
        self.bos = bos
        self.eos = eos

        # Cheat to understand which captions to fetch from the .json file
        self.lang_key = '{}Cap'.format(self.vocab.name)

        with open(self.path) as jf:
            self.data = {}
            self.keys = []
            self.lengths = []
            data = json.load(jf)
            for sample in data:
                # Tile 10 captions over videoIDs
                key = sample['videoID']
                for cidx, capt in enumerate(sample[self.lang_key]):
                    capt = self.vocab.sent_to_idxs(capt, self.bos, self.eos)
                    ckey = '{}#{}'.format(key, cidx)
                    self.data[ckey] = capt
                    self.keys.append(ckey)
                    self.lengths.append(len(capt))

        self.size = len(self.data)

    @staticmethod
    def to_torch(batch):
        return pad_sequence(
            [torch.tensor(b, dtype=torch.long) for b in batch], batch_first=False)

    def __getitem__(self, idx):
        """index is a string with `videoID#captID`."""
        return self.data[idx]

    def __len__(self):
        return self.size

    def __repr__(self):
        s = "{} '{}' ({} captions)\n".format(
            self.__class__.__name__, self.path, self.__len__())
        return s
