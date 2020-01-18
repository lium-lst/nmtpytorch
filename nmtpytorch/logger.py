# -*- coding: utf-8 -*-
import pathlib
import logging

from .cleanup import cleanup


class Logger:
    _logger = None

    def __init__(self, opts=None):
        if Logger._logger is None:
            print('Creating logger')
            _format = '%(message)s'
            _formatter = logging.Formatter(_format)
            Logger._logger = logging.getLogger('nmtpytorch')
            Logger._logger.setLevel(logging.DEBUG)

            con_handler = logging.StreamHandler()
            con_handler.setFormatter(_formatter)
            Logger._logger.addHandler(con_handler)

            if opts is not None:
                log_root = pathlib.Path(opts['save_path'])
                log_root = log_root / opts['subfolder'] / opts['exp_id']
                log_file = str(log_root) + '.log'
                print(f' Will save logs to {log_file}')
                file_handler = logging.FileHandler(log_file, mode='w')
                file_handler.setFormatter(_formatter)
                Logger._logger.addHandler(file_handler)

            cleanup.register_handler(Logger._logger)

    @classmethod
    def log(cls, msg):
        """Plain logging."""
        cls._logger.info(msg)

    def log_prefix(cls, msg, prefix):
        """Plain logging with prefix string."""
        cls._logger.info(f'|{prefix:<30}| {msg}')

    def log_header(cls, msg):
        """Adds a fancy header before and after the message."""
        header_len = len(msg.split('\n')[0])
        header_str = '-' * header_len
        cls._logger.info(f"{header_str}\n{msg}\n{header_str}")
