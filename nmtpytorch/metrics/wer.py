# -*- coding: utf-8 -*-
import editdistance

from .metric import Metric


class WERScorer:
    def compute(self, refs, hyps, language=None, lowercase=False):
        if isinstance(hyps, str):
            # hyps is a file
            hyp_sents = open(hyps).read().strip().split('\n')
        elif isinstance(hyps, list):
            hyp_sents = hyps

        # refs is a list, take its first item
        with open(refs[0]) as f:
            ref_sents = f.read().strip().split('\n')

        assert len(hyp_sents) == len(ref_sents), "WER: # of sentences does not match."

        n_ref_tokens = 0
        dist = 0
        for hyp, ref in zip(hyp_sents, ref_sents):
            hyp_tokens = hyp.split(' ')
            ref_tokens = ref.split(' ')
            n_ref_tokens += len(ref_tokens)
            dist += editdistance.eval(hyp_tokens, ref_tokens)

        score = (100 * dist) / n_ref_tokens
        verbose_score = "{:.3f}% (n_errors = {}, n_ref_tokens = {})".format(
            score, dist, n_ref_tokens)

        return Metric('WER', score, verbose_score, higher_better=False)
