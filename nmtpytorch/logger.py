# -*- coding: utf-8 -*-
from .cleanup import cleanup
import pathlib
import logging


def setup(opts=None):
    _format = '%(message)s'

    formatter = logging.Formatter(_format)
    logger = logging.getLogger('nmtpytorch')
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if opts is not None:
        log_file = str(pathlib.Path(opts['save_path']) /
                       opts['subfolder'] / opts['exp_id']) + '.log'
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    cleanup.register_handler(logger)
    return logger
