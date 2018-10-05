# -*- coding: utf-8 -*-
# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help

import os
import shutil
import threading
import subprocess

from ...utils.misc import get_meteor_jar


class Meteor:
    def __init__(self, language, norm=False):
        self.jar = str(get_meteor_jar())
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', self.jar,
                           '-', '-', '-stdio', '-l', language]
        self.env = os.environ
        self.env['LC_ALL'] = 'en_US.UTF_8'

        # Sanity check
        if shutil.which('java') is None:
            raise RuntimeError('METEOR requires java which is not installed.')

        if norm:
            self.meteor_cmd.append('-norm')

        self.meteor_p = subprocess.Popen(self.meteor_cmd,
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE,
                                         env=self.env,
                                         universal_newlines=True, bufsize=1)
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def method(self):
        return "METEOR"

    def compute_score(self, gts, res):
        imgIds = sorted(list(gts.keys()))
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in imgIds:
            assert len(res[i]) == 1

            hypothesis_str = res[i][0].replace('|||', '').replace('  ', ' ')
            score_line = ' ||| '.join(
                ('SCORE', ' ||| '.join(gts[i]), hypothesis_str))

            # We obtained --> SCORE ||| reference 1 words |||
            # reference n words ||| hypothesis words
            self.meteor_p.stdin.write(score_line + '\n')
            stat = self.meteor_p.stdout.readline().strip()
            eval_line += ' ||| {}'.format(stat)

        # Send to METEOR
        self.meteor_p.stdin.write(eval_line + '\n')

        # Collect segment scores
        for i in range(len(imgIds)):
            score = float(self.meteor_p.stdout.readline().strip())
            scores.append(score)

        # Final score
        final_score = 100 * float(self.meteor_p.stdout.readline().strip())
        self.lock.release()

        return final_score, scores

    def __del__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.wait()
        self.lock.release()
