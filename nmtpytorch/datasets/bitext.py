# -*- coding: utf-8 -*-
from torch.utils.data import Dataset

from ..samplers import BucketBatchSampler
from ..utils.data import read_sentences
from .collate import get_collate


class BitextDataset(Dataset):
    r"""A PyTorch dataset for parallel NMT corpora."""
    def __init__(self, split, data_dict, vocabs, topology,
                 max_trg_len=None, trg_bos=True):
        self.data = {}
        self.lens = {}
        self.split = '%s_set' % split
        self.data_dict = data_dict
        self.vocabs = vocabs
        self.topo = topology
        self.n_sentences = 0
        self.max_trg_len = max_trg_len
        self.trg_bos = trg_bos

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
                raise RuntimeError("Multiple target files not supported.")

            self.data[self.tl], self.lens[self.tl] = \
                read_sentences(fnames[0], self.trg_vocab, bos=self.trg_bos)

            assert len(self.data[self.tl]) == len(self.data[self.sl]), \
                "Number of sentences on both sides differ!"

        # Set keys that will be used by getitem to traverse dataset
        self.data_keys = sorted(list(self.data.keys()))

    def get_loader_args(self, batch_size, drop_targets=False, inference=False):
        """Returns a dictionary with ``DataLoader`` arguments inside.

        Arguments:
            batch_size (int): (Maximum) number of elements in a batch.
            drop_targets (bool, optional): If `True`, batches will not contain
                target-side data even that's available through configuration.
            inference (bool, optional): Should be `True` when doing beam-search.
        """
        keys = self.topo.get_src_langs() if drop_targets else self.data.keys()
        # Create sequence-length ordered sampler
        sampler = BucketBatchSampler(
            batch_size=batch_size, sort_lens=self.lens[self.sl],
            filter_lens=self.lens.get(self.tl, None),
            max_len=self.max_trg_len, store_indices=inference)
        return {
            'batch_sampler': sampler,
            'collate_fn': get_collate(keys),
        }

    def __getitem__(self, idx):
        return {k: self.data[k][idx] for k in self.data_keys}

    def __len__(self):
        return self.size

    def __repr__(self):
        return "[{} {}] ({} - {} samples)".format(
            self.__class__.__name__, self.split,
            self.topo.direction, self.__len__())
