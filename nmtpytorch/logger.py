# -*- coding: utf-8 -*-
import pathlib
import logging

from .cleanup import cleanup


def setup(opts=None):
    _format = '%(message)s'

    formatter = logging.Formatter(_format)
    logger = logging.getLogger('nmtpytorch')
    logger.setLevel(logging.DEBUG)

    con_handler = logging.StreamHandler()
    con_handler.setFormatter(formatter)
    logger.addHandler(con_handler)

    if opts is not None:
        log_file = str(pathlib.Path(opts['save_path']) /
                       opts['subfolder'] / opts['exp_id']) + '.log'
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    cleanup.register_handler(logger)
    return logger
