# -*- coding: utf-8 -*-
import subprocess

from ..utils.misc import listify
from .metric import Metric
import sacrebleu
import logging


class SACREBLEU_MEMScorer:
    """SACREBLEU_MEMScorer class."""
    def __init__(self):
        #self.__cmdline = ["sacrebleu", "--short"]
        # bleu = sacrebleu.corpus_bleu(hyp, ref, lowercase=(case_sensitive==False))
        
        pass

    def compute(self, refs, hyps, language=None, lowercase=False):
        #cmdline = self.__cmdline[:]

        #if lowercase:
        #    cmdline.append("-lc")

        # Make reference files a list
        #cmdline.extend(listify(refs))

        #if isinstance(hyps, str):
        #    hypstring = open(hyps).read().strip()
        #elif isinstance(hyps, list):
        #    hypstring = "\n".join(hyps)
        logging.getLogger("sacrebleu").setLevel(logging.ERROR)
        bleu = sacrebleu.corpus_bleu(hyps, refs, lowercase=lowercase, force=True)

        if bleu is None:
            return Metric('SACREBLEU_MEM', 0, "0.0")
        else:
            score = bleu.score
            float_score = float(score)
            verbose_score = bleu.format()
            return Metric('SACREBLEU_MEM', float_score, verbose_score)
