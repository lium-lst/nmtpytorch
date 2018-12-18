# -*- coding: utf-8 -*-
import pickle as pkl

import torch
from torch import nn


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
