# -*- coding: utf-8 -*-
from collections import OrderedDict

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler

from ..samplers import BucketBatchSampler
from ..utils.data import read_sentences
from .collate import get_collate_v1


class BitextDataset(Dataset):
    r"""A PyTorch dataset for parallel corpora."""
    def __init__(self, split, data_dict, vocabs, topology,
                 max_trg_len=None):
        self.data = {}
        self.lens = {}
        self.split = '%s_set' % split
        self.data_dict = data_dict
        self.vocabs = vocabs
        self.topo = topology
        self.n_sentences = 0
        self.max_trg_len = max_trg_len

        src_langs = self.topo.get_src_langs()
        trg_langs = self.topo.get_trg_langs()

        assert len(src_langs) == len(trg_langs) == 1, \
            "BitextDataset only supports one language at each side."

        self.sl = src_langs[0]
        self.tl = trg_langs[0]

        # Set vocabularies
        self.src_vocab = self.vocabs[self.sl]
        self.trg_vocab = self.vocabs[self.tl]

        #######################
        # Load source sentences
        #######################
        path = self.data_dict[self.split][self.sl]
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
        if self.tl in self.data_dict[self.split]:
            path = self.data_dict[self.split][self.tl]
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
            # Ordered Sampler, translation mode
            sampler = SequentialSampler(self)
            data_loader = DataLoader(
                self, sampler=sampler, shuffle=False, batch_size=batch_size,
                collate_fn=get_collate_v1(self.topo.get_src_langs()))

        else:
            # Training or val perplexity mode, sequence-length ordered sampler
            sampler = BucketBatchSampler(self.lens[self.tl],
                                         batch_size=batch_size,
                                         max_len=self.max_trg_len)
            data_loader = DataLoader(
                self, batch_sampler=sampler,
                collate_fn=get_collate_v1(self.data.keys()))

        return data_loader

    def __getitem__(self, idx):
        return OrderedDict([(k, self.data[k][idx]) for k in self.data_keys])

    def __len__(self):
        return self.size

    def __repr__(self):
        return " [{} {}] ({} - {} samples)".format(
            self.__class__.__name__, self.split,
            self.topo.direction, self.__len__())
