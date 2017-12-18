# -*- coding: utf-8 -*-
from collections import UserDict
import numpy as np

import torch
from torch.autograd import Variable

from ..config import INT
from ..utils.misc import fopen, pbar


def sort_batch(seqbatch):
    """Sorts torch tensor of integer indices by decreasing order."""
    # 0 is padding_idx
    omask = (seqbatch != 0)
    olens = omask.sum(0)
    slens, sidxs = torch.sort(olens, descending=True)
    oidxs = torch.sort(sidxs)[1]
    return (oidxs, sidxs, slens.data.tolist(), omask.float())


def pad_data(seqs):
    """Pads sequences with zero for minibatch processing."""
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)
    out = torch.LongTensor([s + [0] * (max_len - len_) for
                            s, len_ in zip(seqs, lengths)]).t()
    return out


def to_var(input_, requires_grad=False, volatile=False):
    """Returns a torch Variable on GPU."""
    if isinstance(input_, (UserDict, dict)):
        for key in input_:
            v = Variable(input_[key],
                         requires_grad=requires_grad, volatile=volatile)
            input_[key] = v.cuda()
    else:
        input_ = Variable(
            input_, requires_grad=requires_grad, volatile=volatile).cuda()
    return input_


def get_collate_fn(keys):
    """Returns a special collate_fn which will view the underlying data
    in terms of the given keys."""

    def collate_fn(batch):
        tensors = UserDict()
        tensors.size = len(batch)

        # Iterate over data sources
        for key in keys:
            if isinstance(batch[0][key], list):
                # Sequence vocabulary indices
                tensors[key] = pad_data([elem[key] for elem in batch])
            elif isinstance(batch[0][key], np.ndarray):
                # Image data
                tensors[key] = torch.stack([torch.from_numpy(elem[key])
                                            for elem in batch])

        return tensors
    return collate_fn


class CircularNDArray(object):
    def __init__(self, data, n_tile):
        """Access to same elements over first axis using LUT."""
        self.data = data

        # Actual data items size in the first axis
        self.n_elem = self.data.shape[0]

        self._lut = tuple([k % self.n_elem for k
                          in range(self.n_elem * n_tile)])

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, idx):
        # native: 28us
        # only ~2x lower than native access (55 us)
        return self.data[self._lut[idx]]
        # 55 us return self.__dict__['data'][self._lut[idx]]
        # 71us return self.data[idx % self.n_elem]
        # 68 return self.data[idx % 29000]


def read_sentences(fname, vocab, bos=False, slist_max=0, verbose=False):
    lines = []
    lens = []
    with fopen(fname) as f:
        for idx, line in enumerate(pbar(f, unit='sents')):
            line = line.strip()

            # Empty lines will cause a lot of headaches,
            # get rid of them during preprocessing!
            assert line, "Empty line (%d) found in %s" % (idx + 1, fname)

            # Map and append
            seq = vocab.sent_to_idxs(line, limit=slist_max, explicit_bos=bos)
            lines.append(seq)
            lens.append(len(seq))

    return lines, lens
