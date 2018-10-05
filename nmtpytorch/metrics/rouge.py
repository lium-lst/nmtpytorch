# -*- coding: utf-8 -*-
from .metric import Metric
from ..cocoeval import Rouge


class ROUGEScorer:
    def compute(self, refs, hyps, language=None, lowercase=False):
        if isinstance(hyps, str):
            # hyps is a file
            hyp_sents = open(hyps).read().strip().split('\n')
        elif isinstance(hyps, list):
            hyp_sents = hyps

        # refs is a list, take its first item
        with open(refs[0]) as f:
            ref_sents = f.read().strip().split('\n')

        assert len(hyp_sents) == len(ref_sents), "ROUGE: # of sentences does not match."

        rouge_scorer = Rouge()

        rouge_sum = 0
        for hyp, ref in zip(hyp_sents, ref_sents):
            rouge_sum += rouge_scorer.calc_score([hyp], [ref])

        score = (100 * rouge_sum) / len(hyp_sents)
        verbose_score = "{:.3f}".format(score)

        return Metric('ROUGE', score, verbose_score, higher_better=True)
