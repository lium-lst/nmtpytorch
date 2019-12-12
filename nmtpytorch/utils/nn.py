# -*- coding: utf-8 -*-
import pickle as pkl

import torch
from torch import nn
import torch.nn.functional as F


def get_rnn_hidden_state(h):
    """Returns h_t transparently regardless of RNN type."""
    return h if not isinstance(h, tuple) else h[0]


def get_activation_fn(name):
    """Returns a callable activation function from torch."""
    if name in (None, 'linear'):
        return lambda x: x
    elif name in ('sigmoid', 'tanh'):
        return getattr(torch, name)
    else:
        return getattr(F, name)


def mean_pool(data):
    """Simple mean pool function for transforming 3D features of shape
    [T]imesteps x [B]atch_size x [F]eature_size into 2D BxF features.
    (author: @klmulligan)

        Arguments:
            data (tuple): Encoder result of form (data: Tensor(TxBxF), mask: Tensor(TxB))
        Returns:
            pooled_data (Tensor): Mean pooled data of shape BxF.
    """
    # Unpack
    x, mask = data

    if mask is not None:
        return x.sum(0) / mask.sum(0).unsqueeze(1)
    else:
        return x.mean(0)


def get_partial_embedding_layer(vocab, embedding_dim, pretrained_file,
                                freeze='none', oov_zero=True):
    """A partially updateable embedding layer with pretrained embeddings.
    This is experimental and not quite tested."""
    avail_idxs, miss_idxs = [], []
    avail_embs = []

    # Load the pickled dictionary
    with open(pretrained_file, 'rb') as f:
        pret_dict = pkl.load(f)

    for idx, word in vocab._imap.items():
        if word in pret_dict:
            avail_embs.append(pret_dict[word])
            avail_idxs.append(idx)
        else:
            miss_idxs.append(idx)

    # This matrix contains the pretrained embeddings
    avail_embs = torch.Tensor(avail_embs)

    # We don't need the whole dictionary anymore
    del pret_dict

    n_pretrained = len(avail_idxs)
    n_learned = vocab.n_tokens - n_pretrained

    # Sanity checks
    assert len(avail_idxs) + len(miss_idxs) == vocab.n_tokens

    # Create the layer
    emb = nn.Embedding(vocab.n_tokens, embedding_dim, padding_idx=0)
    if oov_zero:
        emb.weight.data.fill_(0)

    # Copy in the pretrained embeddings
    emb.weight.data[n_learned:] = avail_embs
    # Sanity check
    assert torch.equal(emb.weight.data[-1], avail_embs[-1])

    grad_mask = None
    if freeze == 'all':
        emb.weight.requires_grad = False
    elif freeze == 'partial':
        # Create bitmap gradient mask
        grad_mask = torch.ones(vocab.n_tokens)
        grad_mask[n_learned:].fill_(0)
        grad_mask[0].fill_(0)
        grad_mask.unsqueeze_(1)

        def grad_mask_hook(grad):
            return grad_mask.to(grad.device) * grad

        emb.weight.register_hook(grad_mask_hook)

    # Return the layer
    return emb
