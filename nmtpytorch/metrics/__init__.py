# -*- coding: utf-8 -*-
import os
import operator
from collections import OrderedDict

import numpy as np

from .bleu import MultiBleuScorer
from .meteor import METEORScorer
from .mtevalbleu import MTEvalV13aBLEUScorer
from .external import ExternalScorer
from ..utils.filterchain import FilterChain
from ..utils.misc import get_language


def get_scorer(scorer):
    scorers = {'meteor':    METEORScorer,
               'bleu':      MultiBleuScorer,
               'bleu_v13a': MTEvalV13aBLEUScorer}

    if scorer in scorers:
        # A defined metric
        return scorers[scorer]()
    elif scorer.startswith(('/', '~')):
        # External script
        return ExternalScorer(os.path.expanduser(scorer))()


class Evaluator(object):
    comparators = {
        'ter':          (min, operator.lt, -1),
        'loss':         (min, operator.lt, -1),
        'bleu':         (max, operator.gt, 0),
        'cider':        (max, operator.gt, 0),
        'rouge':        (max, operator.gt, 0),
        'meteor':       (max, operator.gt, 0),
        'bleu_v13a':    (max, operator.gt, 0),
    }

    def __init__(self, refs, metrics, filters=''):
        self.scorers = OrderedDict()
        self.kwargs = {}
        self.filter = FilterChain(filters) if filters else lambda s: s
        self.refs = list(refs.parent.glob(refs.name))
        self.language = get_language(self.refs[0])
        if filters:
            # Let FilterChain create copies of reference files with
            # post-processing applied and return their paths
            self.refs = self.filter(refs)

        assert len(self.refs) > 0, "Number of reference files == 0"

        # Convert to list if not
        if isinstance(metrics, str):
            metrics = metrics.split(',')

        for metric in sorted(metrics):
            self.scorers[metric] = get_scorer(metric)
            self.kwargs[metric] = {}
            if metric == 'meteor':
                self.kwargs[metric]['language'] = self.language

    def score(self, hyps):
        """hyps is a list of hypotheses as they come out from decoder."""
        assert isinstance(hyps, list), "hyps should be a list."

        # Post-process if requested
        hyps = self.filter(hyps)

        scores = OrderedDict()
        for key, scorer in self.scorers.items():
            scores[key] = scorer.compute(self.refs, hyps, **self.kwargs[key])
        return scores

    @staticmethod
    def to_file(hyps, fname, vocab=None, n_best=1):
        """Dumps the given hypotheses to a file.

        Arguments:
            hyps(list): List of list of integers if `n_best=1` else
                List of list of lists.
            fname(str): Output file name.
            vocab(Vocabulary, optional): Vocabulary for mapping
                back to tokens. No need to provide if ``hyps`` is already
                a list of string.
            n_best(int, optional): How many hypotheses per sample requested.
                If ``hyps`` is already a list of string, this argument has
                no impact.

        """
        assert isinstance(hyps, list), "hyps should be a list."

        out = open(fname, 'w')

        if isinstance(hyps[0], str):
            for hyp in hyps:
                out.write(hyp + '\n')
        else:
            for i, hyp in enumerate(hyps):
                repr_ = ""
                if len(hyp) == 1:
                    repr_ = vocab.idxs_to_sent(hyp)
                else:
                    # Get scores from first index
                    scores = [(k, s.pop(0)) for k, s in enumerate(hyp)]
                    scores = sorted(scores, key=lambda x: x[0])
                    for idx, score in scores[:n_best]:
                        repr_ += "{} ||| {} ||| {:.4f}\n".format(
                            i, vocab.idxs_to_sent(hyp[idx]), score)

                out.write(repr_ + '\n')
        out.close()

    @staticmethod
    def find_best(name, history):
        """Returns the best idx and value for the given metric."""
        history = np.array(history)
        if name.startswith(('bleu', 'meteor', 'cider', 'rouge')):
            best_idx = np.argmax(history)
        elif name in ['loss', 'px', 'ter']:
            best_idx = np.argmin(history)

        # Validation periods start from 1
        best_val = history[best_idx]
        return (best_idx + 1), best_val

    @staticmethod
    def is_last_best(name, history, min_delta=0.0):
        """Checks whether the last element in history is the best score so far
        by taking into account an absolute improvement threshold min_delta.

        Args:
            name (str): Name of the key for the metric in history dict
            history (dict): A dictionary of metric keys into history list
            min_delta (float, optional): Absolute improvement threshold
                to be taken into account.

        Returns:
            bool.

        """

        # If first validation, return True to save it
        if len(history) == 1:
            return True

        new_value = history[-1]

        # bigger is better
        if name.startswith(('bleu', 'meteor', 'cider', 'rouge')):
            cur_best = max(history[:-1])
            return new_value > cur_best and \
                abs(new_value - cur_best) >= (min_delta - 1e-5)
        # lower is better
        elif name in ['loss', 'px', 'ter']:
            cur_best = min(history[:-1])
            return new_value < cur_best and \
                abs(new_value - cur_best) >= (min_delta - 1e-5)
