# -*- coding: utf-8 -*-
import os
import pathlib
import subprocess
import pkg_resources

from ..utils.misc import listify
from .metric import Metric

METEOR_JAR = pkg_resources.resource_filename('nmtpytorch',
                                             'lib/meteor-1.5.jar')


class METEORScorer(object):
    def __init__(self):
        self.__cmdline = ["java", "-Xmx2G", "-jar", METEOR_JAR,
                          "-", "-", "-stdio"]
        self.env = os.environ
        self.env['LC_ALL'] = 'en_US.UTF-8'

    def compute(self, refs, hyps, language="auto"):
        cmdline = self.__cmdline[:]
        refs = listify(refs)

        if isinstance(hyps, str):
            # If file, open it for line reading
            hyps = open(hyps)

        if language == "auto":
            # Take the extension of the 1st reference file, e.g. ".de"
            language = pathlib.Path(refs[0]).suffix[1:]

        cmdline.extend(["-l", language])

        # Make reference files a list
        iters = [open(f) for f in refs]
        iters.append(hyps)

        # Run METEOR process
        proc = subprocess.Popen(cmdline,
                                stdout=subprocess.PIPE,
                                stdin=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                env=self.env,
                                universal_newlines=True, bufsize=1)

        eval_line = 'EVAL'

        for line_ctr, lines in enumerate(zip(*iters)):
            lines = [l.rstrip('\n') for l in lines]
            refstr = " ||| ".join(lines[:-1])
            line = "SCORE ||| " + refstr + " ||| " + lines[-1]

            proc.stdin.write(line + '\n')
            eval_line += ' ||| {}'.format(proc.stdout.readline().strip())

        # Send EVAL line to METEOR
        proc.stdin.write(eval_line + '\n')

        # Dummy read segment scores
        for i in range(line_ctr + 1):
            proc.stdout.readline().strip()

        # Compute final METEOR
        try:
            score = float(proc.stdout.readline().strip())
            score = Metric('METEOR', 100 * score)
        except Exception as e:
            score = Metric('METEOR', 0.0)
        finally:
            # Close METEOR process
            proc.stdin.close()
            proc.terminate()
            proc.kill()
            proc.wait(timeout=2)
            return score
