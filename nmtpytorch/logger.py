# -*- coding: utf-8 -*-
from .cleanup import cleanup
import pathlib
import logging

log = None


def setup(opts, mode):
    global log
    if log:
        return log

    _format = '%(message)s'

    formatter = logging.Formatter(_format)
    logger = logging.getLogger('nmtpytorch')
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if mode != 'translate':
        log_file = str(pathlib.Path(opts['save_path']) /
                       opts['subfolder'] / opts['exp_id']) + '.log'
        file_mode = 'a' if mode == 'resume' else 'w'

        fh = logging.FileHandler(log_file, mode=file_mode)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    log = logger
    cleanup.register_handler(log)
    return log
