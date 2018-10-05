# -*- coding: utf-8 -*-
import subprocess

from ..utils.misc import listify
from .metric import Metric


class SACREBLEUScorer:
    """SACREBLEUScorer class."""
    def __init__(self):
        self.__cmdline = ["sacrebleu", "--short"]

    def compute(self, refs, hyps, language=None, lowercase=False):
        cmdline = self.__cmdline[:]

        if lowercase:
            cmdline.append("-lc")

        # Make reference files a list
        cmdline.extend(listify(refs))

        if isinstance(hyps, str):
            hypstring = open(hyps).read().strip()
        elif isinstance(hyps, list):
            hypstring = "\n".join(hyps)

        score = subprocess.run(cmdline, stdout=subprocess.PIPE,
                               input=hypstring,
                               universal_newlines=True).stdout.splitlines()

        if len(score) == 0:
            return Metric('SACREBLEU', 0, "0.0")
        else:
            score = score[0].strip()
            float_score = float(score.split()[2])
            verbose_score = ' '.join(score.split()[2:])
            return Metric('SACREBLEU', float_score, verbose_score)
