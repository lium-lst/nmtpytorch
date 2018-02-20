# -*- coding: utf-8 -*-
import torch
from .utils.nn import tile_ctx_dict
from .utils.misc import pbar
from .utils.data import to_var


def beam_search(model, data_loader, vocab, beam_size=12, max_len=200,
                avoid_double=False, avoid_unk=False):
    """An efficient GPU implementation for beam-search algorithm.

    Arguments:
        model (Model): A model instance derived from `nn.Module` defining
            a set of methods. See `models/nmt.py`.
        data_loader (DataLoader): A ``DataLoader`` instance returned by the
            ``get_iterator()`` method of your dataset.
        vocab (Vocabulary): Vocabulary dictionary for the decoded language.
        beam_size (int, optional): The size of the beam. (Default: 12)
        max_len (int, optional): Maximum target length to stop beam-search
            if <eos> is still not generated. (Default: 200)
        avoid_double (bool, optional): Suppresses probability of a token if
            it was already decoded in the previous timestep. (Default: False)
        avoid_unk (bool, optional): Prevents <unk> generation. (Default: False)

    Returns:
        list:
            A list of hypotheses in surface form.
    """

    results = []

    bos = vocab['<bos>']
    eos = vocab['<eos>']
    unk = vocab['<unk>']
    n_vocab = len(vocab)
    inf = 1e3
    k = beam_size

    for batch in pbar(data_loader, unit='batch'):
        n = batch.size

        # Mask to apply to pdxs.view(-1) to fix indices
        nk_mask = torch.arange(n * k).long().cuda()
        pdxs_mask = (nk_mask / k) * k

        # Tile indices to use in the loop to expand first dim
        tile = nk_mask / k

        # Encode source modalities
        ctx_dict = model.encode(to_var(batch, volatile=True))

        # We can fill this to represent the beams in tensor format
        beam = torch.zeros((max_len, n, k)).long().cuda()

        # Get initial decoder state (N*H)
        h_t = model.dec.f_init(*ctx_dict['txt'])

        # Initial y_t for <bos> embs: N x emb_dim
        y_t = model.dec.emb(to_var(
            torch.ones(n).long() * bos, volatile=True))

        log_p, h_t = model.dec.f_next(ctx_dict, y_t, h_t)
        nll, beam[0] = log_p.data.topk(k, sorted=False, largest=False)

        for t in range(1, max_len):
            cur_tokens = beam[t - 1].view(-1)
            fini_idxs = (cur_tokens == eos).nonzero()
            n_fini = fini_idxs.numel()
            if n_fini == n * k:
                break

            # Fetch embs for the next iteration (N*K, E)
            y_t = model.dec.emb(to_var(cur_tokens, volatile=True))

            # Get log_probs and new RNN states (log_p, N*K, V)
            ctx_dict = tile_ctx_dict(ctx_dict, tile)
            log_p, h_t = model.dec.f_next(ctx_dict, y_t, h_t[tile])
            log_p = log_p.data

            # Suppress probabilities of previous tokens
            if avoid_double:
                log_p.view(-1).index_fill_(
                    0, cur_tokens + (nk_mask * n_vocab), inf)

            # Avoid <unk> tokens
            if avoid_unk:
                log_p[:, unk] = inf

            # Favor finished hyps to generate <eos> again
            # Their nll scores will not increase further and they will
            # always be kept in the beam.
            if n_fini > 0:
                fidxs = fini_idxs[:, 0]
                log_p.index_fill_(0, fidxs, inf)
                log_p.view(-1).index_fill_(
                    0, fidxs * n_vocab + eos, 0)

            # Expand to 3D, cross-sum scores and reduce back to 2D
            nll = (nll.unsqueeze(2) + log_p.view(n, k, -1)).view(n, -1)

            # Reduce (N, K*V) to k-best
            nll, idxs = nll.topk(k, sorted=False, largest=False)

            # previous indices into the beam and current token indices
            pdxs = idxs / n_vocab

            # Insert current tokens
            beam[t] = idxs % n_vocab

            # Permute all hypothesis history according to new order
            beam[:t] = beam[:t].gather(2, pdxs.repeat(t, 1, 1))

            # Compute correct previous indices
            # Mask is needed since we're in flattened regime
            tile = pdxs.view(-1) + pdxs_mask

        # Put an explicit <eos> to make idxs_to_sent happy
        beam[max_len - 1] = eos

        # Find lengths by summing tokens not in (pad,bos,eos)
        lens = (beam.transpose(0, 2) > 2).sum(-1).t().float().clamp(min=1)

        # Normalize scores by length
        nll /= lens.float()
        top_hyps = nll.topk(1, sorted=False, largest=False)[1].squeeze(1)

        # Get best hyp for each sample in the batch
        hyps = beam[:, range(n), top_hyps].cpu().numpy().T
        results.extend(vocab.list_of_idxs_to_sents(hyps))

    return results
