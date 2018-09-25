# -*- coding: utf-8 -*-
from collections import namedtuple

import numpy as np
import torch

from .utils.misc import pbar


Hypothesis = namedtuple('Hypothesis', ['tokens', 'score'])


def empty_hypothesis():
    return Hypothesis([], 0.0)


def expand_null(hyp, null_score):
    return Hypothesis(hyp.tokens, hyp.score + null_score)


def expand_token(hyp, token, score):
    return Hypothesis(hyp.tokens + [token], hyp.score + score)


def recombine_hyps(hyp1, hyp2):
    assert hyp1.tokens == hyp2.tokens
    return Hypothesis(hyp1.tokens, np.logaddexp(hyp1.score, hyp2.score))


def tile_ctx_dict(ctx_dict, idxs):
    """Returns dict of 3D tensors repeatedly indexed along the sample axis."""
    # 1st: tensor, 2nd optional mask
    return {
        k: (t[:, idxs], None if mask is None else mask[:, idxs])
        for k, (t, mask) in ctx_dict.items()
    }


def beam_search(models, data_loader, beam_size=12, max_len=200, lp_alpha=0.):
    """An efficient GPU implementation for beam-search algorithm.

    Arguments:
        models (list of Model): Model instance(s) derived from `nn.Module`
            defining a set of methods. See `models/nmt.py`.
        data_loader (DataLoader): A ``DataLoader`` instance.
        beam_size (int, optional): The size of the beam. (Default: 12)
        max_len (int, optional): Maximum target length to stop beam-search
            if <eos> is still not generated. (Default: 200)
        lp_alpha (float, optional): If > 0, applies Google's length-penalty
            normalization instead of simple length normalization.
            lp: ((5 + |Y|)^lp_alpha / (5 + 1)^lp_alpha)

    Returns:
        list:
            A list of hypotheses in surface form.
    """

    # This is the batch-size requested by the user but with sorted
    # batches, efficient batch-size will be <= max_batch_size
    max_batch_size = data_loader.batch_sampler.batch_size
    k = beam_size
    inf = -1000
    results = []
    vocab = models[0].trg_vocab
    n_vocab = len(vocab)

    # Tensorized beam that will shrink and grow upto max_batch_size
    beam_storage = torch.zeros((max_len, max_batch_size, k)).long().cuda()
    mask = torch.arange(max_batch_size * k).long().cuda()
    nll_storage = torch.zeros(max_batch_size).cuda()

    for batch in pbar(data_loader, unit='batch'):
        # Send to GPU
        batch.to_gpu(volatile=True)

        # Always use the initial storage
        beam = beam_storage.narrow(1, 0, batch.size).zero_()

        # Mask to apply to pdxs.view(-1) to fix indices
        nk_mask = mask.narrow(0, 0, batch.size * k)

        # nll: batch_size x 1 (will get expanded further)
        nll = nll_storage.narrow(0, 0, batch.size).unsqueeze(1)

        # Tile indices to use in the loop to expand first dim
        tile = range(batch.size)

        # Encode source modalities
        ctx_dicts = [m.encode(batch) for m in models]

        # Get initial decoder state (N*H)
        h_ts = [m.dec.f_init(ctx_dict) for m, ctx_dict in zip(models, ctx_dicts)]

        ctc_logprobs = m.dec(ctx_dicts, y=None)['logp']

        if beam_size == 1:
            maxima = ctc_logprobs.argmax(dim=2).transpose(0, 1) - 1
            hyps = [out[out != -1] for out in maxima]
        else:
            hyps = []
            np_logprobs = ctc_logprobs.transpose(0, 1).numpy()
            for output in np_logprobs:
                beam = [empty_hypothesis()]
                for step in output:
                    new_beam = []
                    beam_dict = {str(h.tokens): h for h in beam}
                    for hyp in ctc_hyps:
                        # expand with blank first
                        new_beam.append(expand_null(hyp, output[0]))

                        # try to epand with others
                        for token, logprob in enumerate(output):
                            new_hyp = expand_token(hyp, token, logprob)
                            if str(new_hyp.tokens) in beam_dict:
                                parent = beam_dict[str(new_hyp.tokens)]
                                new_beam.append(recombine_hyps(parent, new_hyp))
                            else:
                                new_beam.append(new_hyp)
                new_beam_scores = np.array(h.score for h in new_beam)
                best_hyp_indices = np.argpartition(
                    -new_beam_scores, beam_size)[:beam_size]
                beam = [new_beam[i] for i in best_hyp_indices]
            best = max(beam, key=lambda h: h.score)
            hyps.append(best.tokens)

        results.extend(vocab.list_of_idxs_to_sents(hyps))

    # Recover order of the samples if necessary
    if getattr(data_loader.batch_sampler, 'store_indices', False):
        results = [results[i] for i, j in sorted(
            enumerate(data_loader.batch_sampler.orig_idxs), key=lambda k: k[1])]
    return results
