# -*- coding: utf-8 -*-
import time
import logging
from pathlib import Path

from .utils.misc import load_pt_file

from . import models
from .config import Options

logger = logging.getLogger('nmtpytorch')


class Tester(object):
    """Tester for models without beam-search."""

    def __init__(self, **kwargs):
        # Store attributes directly. See bin/nmtpy for their list.
        self.__dict__.update(kwargs)

        # How many models?
        if len(self.models) > 1:
            raise RuntimeError("Test mode requires single model file.")

        self.model_file = self.models[0]

        weights, _, opts = load_pt_file(self.model_file)
        opts = Options.from_dict(opts)
        instance = getattr(models, opts.train['model_type'])(opts=opts)

        if instance.supports_beam_search:
            logger.info("Model supports beam-search by the way.")

        # Setup layers
        instance.setup(is_train=False)
        # Load weights
        instance.load_state_dict(weights, strict=True)
        # Move to GPU
        instance.cuda()
        # Switch to eval mode
        instance.train(False)

        self.instance = instance

        # Can be a comma separated list of hardcoded test splits
        if self.splits:
            logger.info('Will process "{}"'.format(self.splits))
            self.splits = self.splits.split(',')
        elif self.source:
            # Split into key:value's and parse into dict
            input_dict = {}
            logger.info('Will process input configuration:')
            for data_source in self.source.split(','):
                key, path = data_source.split(':', 1)
                input_dict[key] = Path(path)
                logger.info(' {}: {}'.format(key, input_dict[key]))
            self.instance.opts.data['new_set'] = input_dict
            self.splits = ['new']

    def test(self, instance, split):
        instance.load_data(split)
        loader = instance.datasets[split].get_iterator(
            self.batch_size, inference=True)

        logger.info('Starting computation')
        start = time.time()
        results = instance.test_performance(
            loader,
            dump_file="{}.{}".format(self.model_file, split))
        up_time = time.time() - start
        logger.info('Took {:.3f} seconds'.format(up_time))
        return results

    def __call__(self):
        for input_ in self.splits:
            results = self.test(self.instance, input_)
            for res in results:
                print('  {}: {:.5f}'.format(res.name, res.score))
