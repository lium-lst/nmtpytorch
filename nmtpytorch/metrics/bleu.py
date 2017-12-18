# -*- coding: utf-8 -*-
import subprocess
import pkg_resources

from ..utils.misc import listify
from .metric import Metric

BLEU_SCRIPT = pkg_resources.resource_filename('nmtpytorch',
                                              'lib/multi-bleu.perl')


class MultiBleuScorer(object):
    """MultiBleuScorer class."""
    def __init__(self, lowercase=False):
        # For multi-bleu.perl we give the reference(s) files as argv,
        # while the candidate translations are read from stdin.
        self.lowercase = lowercase
        self.__cmdline = [BLEU_SCRIPT]
        if self.lowercase:
            self.__cmdline.append("-lc")

    def compute(self, refs, hyps):
        cmdline = self.__cmdline[:]

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
            return Metric('BLEU', 0, "0.0")
        else:
            score = score[0].strip()
            float_score = float(score.split()[2][:-1])
            verbose_score = score.replace('BLEU = ', '')
            return Metric('BLEU', float_score, verbose_score)
