import bz2
import gzip
import lzma
import pathlib

import numpy as np
import torch

from ..utils.misc import pbar


def fopen(filename, key=None):
    """gzip,bzip2,xz,numpy aware file opening function."""
    assert '*' not in str(filename), "Glob patterns not supported in fopen()"

    filename = str(pathlib.Path(filename).expanduser())
    if filename.endswith('.gz'):
        return gzip.open(filename, 'rt')
    elif filename.endswith('.bz2'):
        return bz2.open(filename, 'rt')
    elif filename.endswith(('.xz', '.lzma')):
        return lzma.open(filename, 'rt')
    elif filename.endswith(('.npy', '.npz')):
        if filename.endswith('.npz'):
            assert key is not None, "No key= given for .npz file."
            return np.load(filename)[key]
        else:
            return np.load(filename)
    else:
        # Plain text
        return open(filename, 'r')


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


def convert_to_onehot(idxs, n_classes):
    """Returns a binary batch_size x n_classes one-hot tensor."""
    out = torch.zeros(len(idxs), n_classes, device=idxs[0].device)
    for row, indices in zip(out, idxs):
        row.scatter_(0, indices, 1)
    return out


def read_sentences(fname, vocab, bos=False, eos=True):
    lines = []
    lens = []
    basename = pathlib.Path(fname).name
    with fopen(fname) as f:
        for idx, line in enumerate(pbar(f, unit='sents', desc=f'Reading {basename}')):
            line = line.strip()

            # Empty lines will cause a lot of headaches,
            # get rid of them during preprocessing!
            assert line, f"Empty line ({idx + 1}) found in {fname}"

            # Map and append
            seq = vocab.sent_to_idxs(line, explicit_bos=bos, explicit_eos=eos)
            lines.append(seq)
            lens.append(len(seq))

    return lines, lens
