# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path
from collections import defaultdict

import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger('nmtpytorch')


class COCOJSONDataset(Dataset):
    r"""A PyTorch dataset for COCO-style JSON files.

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

        if not self.path.exists():
            raise RuntimeError('{} does not exist.'.format(self.path))

        with open(self.path) as jf:
            data = json.load(jf)
            self.data,  _ = data['annotations'], data['images']

        # Number of captions = dataset size
        self.size = len(self.data)
        self.keys = {}
        self.lengths = []

        # Multiple image_id keys will exist for multiple captions
        # Here we neatly add captionIDs to keep them separate
        next_image_ids = defaultdict(int)

        # Split into words
        for idx, elem in enumerate(self.data):
            # Unpack
            image_id, caption = elem['image_id'], elem['caption']
            next_image_ids[image_id] += 1
            cap_key = '{}@@{}'.format(image_id, next_image_ids[image_id])
            self.keys[idx] = cap_key

            # Map to word indices and insert
            caption = self.vocab.sent_to_idxs(caption, bos, eos)
            self.data[cap_key] = caption

            # Store the length
            self.lengths.append(len(caption))

    @staticmethod
    def to_torch(batch):
        return pad_sequence(
            [torch.tensor(b, dtype=torch.long) for b in batch], batch_first=False)

    def __getitem__(self, idx):
        return self.data[self.keys[idx]]

    def __len__(self):
        return self.size

    def __repr__(self):
        s = "{} '{}' ({} sentences)\n".format(
            self.__class__.__name__, self.fnames[0].name, self.__len__())
        return s
