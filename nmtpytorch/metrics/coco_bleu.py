# -*- coding: utf-8 -*-
import json
import subprocess

from ..utils.misc import get_temp_file
from .metric import Metric


class COCOBLEUScorer:
    """Wrapper to Python 3 cocoeval tools for Bleu_4 evaluation."""
    def __init__(self):
        self.__cmdline = [
            '/data/ozan/git/coco-caption-ozan/cocoeval',
            '-c', 'bleu']
        self.__tmp_ref = get_temp_file()
        self.__tmp_ref.close()

    def compute(self, refs, hyps, language='en'):
        cmdline = self.__cmdline[:]

        # Add refs file
        cmdline.extend(['-g', str(refs[0])])

        # Add results file
        with open(self.__tmp_ref.name, 'w') as f:
            json.dump(hyps, f)
        cmdline.extend(['-r', self.__tmp_ref.name])

        score = subprocess.run(cmdline, stdout=subprocess.PIPE,
                               universal_newlines=True).stdout.splitlines()

        if len(score) == 0:
            return Metric('COCOBLEU', 0, "0.0")
        else:
            score = float(score[0].strip())
            return Metric('COCOBLEU', 100 * score)
