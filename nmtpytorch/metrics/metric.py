# -*- coding: utf-8 -*-
from functools import total_ordering


@total_ordering
class Metric(object):
    def __init__(self, name, score=0.0, detailed_score="", higher_better=True):
        self.name = name
        self.score = score
        self.detailed_score = detailed_score
        self.higher_better = higher_better

    def __eq__(self, other):
        return self.score == other.score

    def __lt__(self, other):
        return self.score < other.score

    def better(self, other):
        less_than_other = self.__lt__(other)
        if not self.higher_better:
            # lower is better
            return less_than_other
        else:
            return not less_than_other

    def __repr__(self):
        if self.detailed_score:
            return "%s = %s" % (self.name, self.detailed_score)
        else:
            return "%s = %.2f" % (self.name, self.score)
