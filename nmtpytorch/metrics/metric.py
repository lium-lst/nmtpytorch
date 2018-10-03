# -*- coding: utf-8 -*-
from functools import total_ordering


@total_ordering
class Metric:
    """A Metric object to represent evaluation metrics.

    Arguments:
        name(str): A name for the metric that will be kept internally
            after upper-casing
        score(float): A floating point score
        detailed_score(str, optional): A custom, more detailed string
            representing the score given above (Default: "")
        higher_better(bool, optional): If ``False``, the smaller the better
            (Default: ``True``)
    """

    def __init__(self, name, score, detailed_score="", higher_better=True):
        self.name = name.upper()
        self.score = score
        self.detailed_score = detailed_score
        self.higher_better = higher_better

    def __eq__(self, other):
        return self.score == other.score

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        rhs = (self.detailed_score if self.detailed_score
               else "%.2f" % self.score)
        return self.name + ' = ' + rhs
