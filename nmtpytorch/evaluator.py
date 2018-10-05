# -*- coding: utf-8 -*-
from collections import OrderedDict

from . import metrics
from .utils.filterchain import FilterChain
from .utils.misc import get_language


class Evaluator:
    def __init__(self, refs, beam_metrics, filters=''):
        # metrics: list of upper-case beam-search metrics
        self.kwargs = {}
        self.scorers = OrderedDict()
        self.refs = list(refs.parent.glob(refs.name))
        self.language = get_language(self.refs[0])
        if self.language is None:
            # Fallback to en (this is only relevant for METEOR)
            self.language = 'en'

        self.filter = lambda s: s
        if filters:
            self.filter = FilterChain(filters)
            self.refs = self.filter(refs)

        assert len(self.refs) > 0, "Number of reference files == 0"

        for metric in sorted(beam_metrics):
            self.kwargs[metric] = {'language': self.language}
            self.scorers[metric] = getattr(metrics, metric + 'Scorer')()

    def score(self, hyps):
        """hyps is a list of hypotheses as they come out from decoder."""
        assert isinstance(hyps, list), "hyps should be a list."

        # Post-process if requested
        hyps = self.filter(hyps)

        results = []
        for key, scorer in self.scorers.items():
            results.append(
                scorer.compute(self.refs, hyps, **self.kwargs[key]))
        return results
