# -*- coding: utf-8 -*-
from collections import UserDict

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ..utils.misc import fopen, pbar


def make_dataloader(dataset, batch_size, pin_memory=False,
                    num_workers=0, **kwargs):
    sampler, collate = dataset.get_sampler(batch_size, **kwargs)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate,
        pin_memory=pin_memory, num_workers=num_workers)


def sort_batch(seqbatch):
    """Sorts torch tensor of integer indices by decreasing order."""
    # 0 is padding_idx
    omask = (seqbatch != 0).long()
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


def onehot_data(idxs, n_classes):
    """Returns a binary batch_size x n_classes one-hot tensor."""
    out = torch.zeros(len(idxs), n_classes)
    for row, indices in zip(out, idxs):
        row.scatter_(0, indices, 1)
    return out


def to_var(input_, requires_grad=False, volatile=False):
    """Returns a torch Variable on GPU."""
    if isinstance(input_, (UserDict, dict)):
        for key in input_:
            input_[key] = Variable(
                input_[key],
                requires_grad=requires_grad, volatile=volatile).cuda(async=False)
        return input_
    else:
        return Variable(
            input_, requires_grad=requires_grad, volatile=volatile).cuda(async=False)


def read_sentences(fname, vocab, bos=False, eos=True):
    lines = []
    lens = []
    with fopen(fname) as f:
        for idx, line in enumerate(pbar(f, unit='sents')):
            line = line.strip()

            # Empty lines will cause a lot of headaches,
            # get rid of them during preprocessing!
            assert line, "Empty line (%d) found in %s" % (idx + 1, fname)

            # Map and append
            seq = vocab.sent_to_idxs(line, explicit_bos=bos, explicit_eos=eos)
            lines.append(seq)
            lens.append(len(seq))

    return lines, lens
