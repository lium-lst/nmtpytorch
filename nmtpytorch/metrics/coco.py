# -*- coding: utf-8 -*-
import json
import subprocess

from ..utils.misc import get_temp_file
from .metric import Metric


class COCOScorer:
    """A generic wrapper for cocoeval script. This class is not meant to
    be used directly, instead should be derived from actual metrics."""
    def __init__(self, metric_param, scale=100):
        self.__cmdline = [
            '/data/ozan/git/coco-caption-ozan/cocoeval',
            '-c', metric_param,
        ]
        self.__pretty_name = f"COCO{metric_param.upper()}"
        self.__scale = scale
        self.__tmp_ref = get_temp_file(close=True)

    def compute(self, refs, hyps, language='en'):
        cmdline = self.__cmdline[:]

        # Add refs file
        cmdline.extend(['-g', str(refs[0])])

        # Add results file
        with open(self.__tmp_ref.name, 'w') as f:
            json.dump(hyps, f)
        cmdline.append(self.__tmp_ref.name)

        score = subprocess.run(
            cmdline, stdout=subprocess.PIPE,
            universal_newlines=True).stdout.splitlines()

        if len(score) == 0:
            return Metric(self.__pretty_name, 0, "0.0")
        else:
            score = float(score[0].strip())
            return Metric(self.__pretty_name, self.__scale * score)


class COCOMETEORScorer(COCOScorer):
    def __init__(self):
        super().__init__(metric_param='meteor', scale=100)


class COCOBLEUScorer(COCOScorer):
    def __init__(self):
        super().__init__(metric_param='bleu', scale=100)


class COCOROUGEScorer(COCOScorer):
    def __init__(self):
        super().__init__(metric_param='rouge', scale=100)


class COCOCIDERScorer(COCOScorer):
    def __init__(self):
        super().__init__(metric_param='cider', scale=10)
