# -*- coding: utf-8 -*-
from collections import OrderedDict

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler

from ..config import FLOAT
from ..samplers import BucketBatchSampler
from ..utils.data import get_collate_fn, read_sentences, CircularNDArray
from ..utils.nn import normalize_images
from ..utils.misc import fopen


class Multi30kDataset(Dataset):
    r"""A PyTorch dataset for Multi30k.

    Arguments:
        split(str): One of the keys with '_set' suffix in [data] section
            of the configuration file.
        img_mode(str): 'raw' for NCHW file (uint8 .npz), a layer string for
            NCHW convolutional feature file (float .npz)
        data_dict(dict): [data] dictionary of the configuration file
        vocabs(dict): dictionary mapping lang keys to Vocabulary() objects
        topology(dict): dictionary defining to src<->trg network topology
        img_norm(bool, optional): When ``img_mode==raw``, normalize raw
            images w.r.t ImageNet statistics. Default: `True`
        logger(Logger): Logger instance to allow printing information

    """
    def __init__(self, split, img_mode, data_dict, vocabs,
                 topology, img_norm=True, logger=None):

        self.data = {}
        self.lens = {}

        # image .npz file key
        self.img_split = split
        self.txt_split = '%s_set' % split

        self.img_mode = img_mode
        self.data_dict = data_dict
        self.vocabs = vocabs
        self.topo = topology
        self.img_norm = img_norm
        self.n_images = 0
        self.n_sentences = 0

        # Setup verbose logging
        self.verbose = logger is not None
        self.print = print if logger is None else logger.info

        self.sl = None if len(self.topo['src_langs']) == 0 else \
            self.topo['src_langs'][0]
        self.tl = self.topo['trg_langs'][0]

        # Set vocabularies
        self.src_vocab = self.vocabs.get(self.sl, None)
        self.trg_vocab = self.vocabs.get(self.tl, None)

        ##############################
        # Load source sentences if any
        ##############################
        if self.sl is not None:
            path = self.data_dict[self.txt_split][self.sl]
            fnames = sorted(path.parent.glob(path.name))
            assert len(fnames) == 1, "Multiple source files not supported."

            self.data[self.sl], self.lens[self.sl] = \
                read_sentences(fnames[0], self.src_vocab)
            self.n_sentences = len(self.data[self.sl])

        ##############################
        # Load target sentences if any
        ##############################
        if self.tl is not None:
            path = self.data_dict[self.txt_split][self.tl]
            fnames = sorted(path.parent.glob(path.name))
            if len(fnames) > 1:
                self.print('Found {} target files, '
                           'using {}'.format(len(fnames), fnames[0].name))

            # <BOS> will be prepended to target sentences
            # Store target sentence lengths for BucketSampler
            self.data[self.tl], self.lens[self.tl] = \
                read_sentences(fnames[0], self.trg_vocab, bos=True)
            if self.n_sentences == 0:
                self.n_sentences = len(self.data[self.tl])
            else:
                assert(self.n_sentences) == len(self.data[self.tl]), \
                    "Sentence count mismatch."

        #############
        # Load images
        #############
        self.print('Loading image data (mode: {})'.format(self.img_mode))
        img_data = fopen(self.data_dict['image'],
                         key=self.img_split).astype(FLOAT)
        self.n_images = img_data.shape[0]

        if self.img_mode == 'raw' and self.img_norm:
            # scale to 0-1, normalize with imagenet channel means
            self.print('Normalizing images using ImageNet statistics')
            normalize_images(img_data)
        elif self.img_mode != 'raw':
            # Flatten spatial dims and transpose
            img_data.shape = (img_data.shape[0], img_data.shape[1], -1)
            img_data = np.transpose(img_data, (0, 2, 1))
            self.print('Shape -> {}'.format(img_data.shape))

        # Decide on dataset size based on splits
        # Duplicate images or features to match dataset size
        # in the case where sentences/image > 1
        self.sent_per_image = self.n_sentences / self.n_images
        assert self.sent_per_image == int(self.sent_per_image), \
            "Number of sentences/image is not an integer."
        self.sent_per_image = int(self.sent_per_image)

        # Dummy circular view over image tensor if multiple sentences/image
        self.data['image'] = img_data if self.sent_per_image == 1 else \
            CircularNDArray(img_data, self.sent_per_image)

        # Set keys that will be used by getitem to traverse dataset
        self.data_keys = sorted(list(self.data.keys()))

        # Dataset size is determined by the largest number of samples
        self.size = max(self.n_sentences, self.n_images)

    def get_iterator(self, batch_size, only_source=False):
        """Returns a DataLoader instance with or without target data."""
        if only_source:
            # Translation mode
            _collate_fn = get_collate_fn(self.topo['srcs'])

            # Ordered Sampler
            sampler = SequentialSampler(self)
            data_loader = DataLoader(self, sampler=sampler,
                                     shuffle=False, batch_size=batch_size,
                                     collate_fn=_collate_fn)

        else:
            # Training mode
            _collate_fn = get_collate_fn(self.data.keys())

            # Sequence-length ordered sampler
            sampler = BucketBatchSampler(self.lens[self.tl],
                                         batch_size=batch_size)
            data_loader = DataLoader(self, batch_sampler=sampler,
                                     collate_fn=_collate_fn)

        return data_loader

    def __getitem__(self, idx):
        return OrderedDict([(k, self.data[k][idx]) for k in self.data_keys])

    def __len__(self):
        return self.size

    def __repr__(self):
        direction = "{}->{}".format(",".join(self.topo['srcs']),
                                    ",".join(self.topo['trgs']))
        return "Multi30kDataset [{}] ({} samples, {} sents / image)".format(
            direction, self.__len__(), self.sent_per_image)
