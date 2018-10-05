# -*- coding: utf-8 -*-
import torch
import logging
from torch.utils.data import DataLoader
import numpy as np

from ..utils.misc import fopen, pbar

logger = logging.getLogger('nmtpytorch')


def make_dataloader(dataset, pin_memory=False, num_workers=0):
    if num_workers != 0:
        logger.info('Forcing num_workers to 0 since it fails with torch 0.4')
        num_workers = 0

    return DataLoader(
        dataset, batch_sampler=dataset.sampler,
        collate_fn=dataset.collate_fn,
        pin_memory=pin_memory, num_workers=num_workers)


def sort_batch(seqbatch):
    """Sorts torch tensor of integer indices by decreasing order."""
    # 0 is padding_idx
    omask = (seqbatch != 0).long()
    olens = omask.sum(0)
    slens, sidxs = torch.sort(olens, descending=True)
    oidxs = torch.sort(sidxs)[1]
    return (oidxs, sidxs, slens.data.tolist(), omask.float())


def pad_video_sequence(seqs):
    """
    Pads video sequences with zero vectors for minibatch processing.
    (contributor: @elliottd)

    TODO: Can we write the for loop in a more compact format?
    """
    lengths = [len(s) for s in seqs]
    # Get the desired size of the padding vector from the input seqs data
    feat_size = seqs[0].shape[1]
    max_len = max(lengths)
    tmp = []
    for s, len_ in zip(seqs, lengths):
        if max_len - len_ == 0:
            tmp.append(s)
        else:
            inner_tmp = s
            for i in range(max_len - len_):
                inner_tmp = np.vstack((inner_tmp, (np.array([0.] * feat_size))))
            tmp.append(inner_tmp)
    padded = np.array(tmp, dtype='float32')
    return torch.FloatTensor(torch.from_numpy(padded))


def onehot_data(idxs, n_classes):
    """Returns a binary batch_size x n_classes one-hot tensor."""
    out = torch.zeros(len(idxs), n_classes)
    for row, indices in zip(out, idxs):
        row.scatter_(0, indices, 1)
    return out


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
