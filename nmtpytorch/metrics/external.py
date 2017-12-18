# -*- coding: utf-8 -*-
import subprocess

from .metric import Metric


class ExternalScore(Metric):
    def __init__(self, score=None):
        super(ExternalScore, self).__init__(score)
        # This should be overriden once the score is received
        # So the script should behave exactly as it is documented below.
        self.name = 'External'

        if score:
            # Parse score line of format:
            # METRIC = SCORE, ......
            name, rest = score.split('=', 1)
            self.score_str = rest.strip()
            self.name = name.strip()
            score, rest = rest.split(',', 1)
            self.score = float(score.strip())


class ExternalScorer(object):
    """An external scorer that calls arbitrary script for metric computation.
        - The script should be runnable as it-is
        - It should consume the hypotheses from stdin and receive
          a variable number of references as cmdline arguments.
        - The script should output a "single line" to stdout with the
          following format:
            METRICNAME = SCORE, <any arbitrary score information string>

        Example:
            $ custombleu.perl ref1 ref2 ref3 < hyps (Higher better)
                BLEU = 23.43, (ref_len=xxx,hyp_len=xxx,penalty=xxx)
            $ wer.py ref1 < hyps (Lower better)
                WER = 32.42, (....)

    """
    def __init__(self, script):
        self.__cmdline = [script]

    def compute(self, refs, hypfile):
        cmdline = self.__cmdline[:]

        # Make reference files a list and add to command
        refs = [refs] if isinstance(refs, str) else refs
        cmdline.extend(refs)

        # Read hypotheses
        with open(hypfile, "r") as fhyp:
            hypstring = fhyp.read().rstrip()

        # Run script
        score = subprocess.run(cmdline, stdout=subprocess.PIPE,
                               input=hypstring,
                               universal_newlines=True).stdout.splitlines()
        if len(score) == 0:
            return ExternalScore()
        else:
            return ExternalScore(score[0].strip())
