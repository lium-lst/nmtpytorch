# -*- coding: utf-8 -*-
from collections import OrderedDict

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler

from ..samplers import BucketBatchSampler
from ..utils.data import get_collate_fn, read_sentences


class BitextDataset(Dataset):
    r"""A PyTorch dataset for parallel corpora."""
    def __init__(self, split, data_dict, vocabs, topology,
                 logger=None, max_trg_len=None, drop_last=False):
        self.data = {}
        self.lens = {}
        self.txt_split = '%s_set' % split
        self.data_dict = data_dict
        self.vocabs = vocabs
        self.topo = topology
        self.n_sentences = 0
        self.max_trg_len = max_trg_len
        self.drop_last = drop_last

        # Setup verbose logging
        self.verbose = logger is not None
        self.print = print if logger is None else logger.info

        assert self.topo['n_src_langs'] == self.topo['n_trg_langs'] == 1, \
            "BitextDataset only supports one language at each side."

        self.sl = self.topo['src_langs'][0]
        self.tl = self.topo['trg_langs'][0]

        # Set vocabularies
        self.src_vocab = self.vocabs[self.sl]
        self.trg_vocab = self.vocabs[self.tl]

        #######################
        # Load source sentences
        #######################
        path = self.data_dict[self.txt_split][self.sl]
        fnames = sorted(path.parent.glob(path.name))
        if len(fnames) == 0:
            raise RuntimeError('{} does not exist.'.format(path))
        elif len(fnames) > 1:
            raise RuntimeError("Multiple source files not supported.")

        self.data[self.sl], self.lens[self.sl] = \
            read_sentences(fnames[0], self.src_vocab)

        self.size = len(self.data[self.sl])

        #######################
        # Load target sentences
        #######################
        if self.tl in self.data_dict[self.txt_split]:
            path = self.data_dict[self.txt_split][self.tl]
            fnames = sorted(path.parent.glob(path.name))
            if len(fnames) == 0:
                raise RuntimeError('{} does not exist.'.format(path))
            elif len(fnames) > 1:
                raise RuntimeError("Multiple source files not supported.")

            self.data[self.tl], self.lens[self.tl] = \
                read_sentences(fnames[0], self.trg_vocab, bos=True)

            assert len(self.data[self.tl]) == len(self.data[self.sl]), \
                "Number of sentences on both sides differ!"

        # Set keys that will be used by getitem to traverse dataset
        self.data_keys = sorted(list(self.data.keys()))

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
                                         batch_size=batch_size,
                                         max_len=self.max_trg_len,
                                         drop_last=self.drop_last)
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
        return "BitextDataset [{}] ({} samples)".format(
            direction, self.__len__())
