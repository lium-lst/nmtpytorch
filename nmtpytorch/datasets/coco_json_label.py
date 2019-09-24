# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path

import torch

from torch.utils.data import Dataset

from ..utils.data import convert_to_onehot

logger = logging.getLogger('nmtpytorch')


class COCOJSONLabelDataset(Dataset):
    r"""A PyTorch dataset for COCO-style JSON files where captions are
    actually space-separated list of labels for multi-label classification.

    Arguments:
        fname (str or Path): A string or ``pathlib.Path`` object giving
            the corpus.
        vocab (Vocabulary): A ``Vocabulary`` instance for the given corpus.
    """

    def __init__(self, fname, vocab, **kwargs):
        self.path = Path(fname)
        self.vocab = vocab

        if not self.path.exists():
            raise RuntimeError('{} does not exist.'.format(self.path))

        # Read COCO file
        with open(self.path) as jf:
            data = json.load(jf)
            annotations,  _ = data['annotations'], data['images']

        # Number of captions = dataset size
        self.data = {}
        self.lengths = None
        self.size = len(annotations)

        # Split into words
        for idx, elem in enumerate(annotations):
            # Unpack
            image_id, caption = elem['image_id'], elem['caption']

            # Map to word indices and insert
            caption = self.vocab.sent_to_idxs(caption, False, False)
            self.data[image_id] = torch.LongTensor(caption)

        self.keys = sorted(self.data.keys())

    @staticmethod
    def to_torch(batch, **kwargs):
        # Hack: kwargs is filled by model through injection into DataSource
        # We always have to have a time/sequence dim at 0
        return convert_to_onehot(batch, kwargs['n_labels']).unsqueeze(0)

    def __getitem__(self, idx):
        return self.data[self.keys[idx]]

    def __len__(self):
        return self.size

    def sample_to_dict(self, idx, cap):
        orig_id = self.keys[idx].split('@@')[0]
        return {'image_id': orig_id, 'caption': cap}

    def __repr__(self):
        s = "{} '{}' ({} sentences)\n".format(
            self.__class__.__name__, self.path.name, self.__len__())
        return s
