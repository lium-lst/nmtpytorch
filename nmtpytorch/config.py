# -*- coding: utf-8 -*-
import copy
import pathlib

from collections import defaultdict

from configparser import ConfigParser
from ast import literal_eval

# Default data types
INT = 'int64'
FLOAT = 'float32'


TRAIN_DEFAULTS = {
    'seed': 1234,            # RNG seed
    'gclip': 5.,             # Clip gradients above clip_c
    'l2_reg': 0.,            # L2 penalty factor
    'patience': 10,          # Early stopping patience
    'optimizer': 'adam',     # adadelta, sgd, rmsprop, adam
    'lr': 0,                 # 0: Use default lr if not precised further
    'device_id': 'auto_1',   # auto_N for automatic N gpus
                             # 0,1,2 for manual N gpus
                             # 0 for 0th (single) GPU
    'disp_freq': 30,         # Training display frequency (/batch)
    'batch_size': 32,        # Training batch size
    'max_epochs': 100,       # Max number of epochs to train
    'eval_beam': 6,          # Validation beam_size
    'eval_freq': 0,          # 0: End of epochs
    'eval_start': 1,         # Epoch which validation will start
    'save_best_n': 4,        # Store a set of 4 best validation models
    'eval_metrics': 'bleu',  # comma sep. metrics, 1st -> earlystopping
    'eval_filters': '',      # comma sep. filters to apply to refs and hyps
    'checkpoint_freq': 0,    # Checkpoint frequency for resuming
    'max_iterations': int(1e6),     # Max number of updates to train
    'patience_delta': 0.,           # Abs. difference that counts for metrics
    'tensorboard_dir': '',          # Enable TB and give global log folder
}


def resolve_path(value):
    if isinstance(value, list):
        return [resolve_path(elem) for elem in value]
    elif isinstance(value, dict):
        return {k: resolve_path(v) for k, v in value.items()}
    elif isinstance(value, str) and value.startswith(('~', '/', '../', './')):
        return pathlib.Path(value).expanduser()
    else:
        return value


class Options(object):
    @staticmethod
    def __parse_value(value):
        """Automatic type conversion for configuration values.

        Arguments:
            value(str): A string to parse.
        """

        # Check for boolean or None
        if str(value).capitalize().startswith(('False', 'True', 'None')):
            return eval(str(value).capitalize(), {}, {})

        else:
            # Detect strings, floats and ints
            try:
                # If this fails, this is a string
                result = literal_eval(value)
            except Exception as ve:
                result = value

            return result

    @classmethod
    def from_dict(cls, dict_):
        """Loads object from dict."""
        obj = cls.__new__(cls)
        obj.__dict__.update(dict_)
        return obj

    def __init__(self, filename, overrides=None):
        self.__parser = ConfigParser()
        self.filename = filename
        self.overrides = defaultdict(dict)
        self.sections = []

        # Read the defaults first
        self.__parser.read_dict({'train': TRAIN_DEFAULTS})

        if len(self.__parser.read(self.filename)) == 0:
            raise Exception('Could not parse configuration file.')

        if overrides is not None:
            # ex: train.batch_size:32
            for opt in overrides:
                section, keyvalue = opt.split('.', 1)
                key, value = keyvalue.split(':')
                value = resolve_path(value)
                self.overrides[section][key] = self.__parse_value(value)

        for section in self.__parser.sections():
            opts = {}
            self.sections.append(section)

            for key, value in self.__parser[section].items():
                opts[key] = resolve_path(self.__parse_value(value))

            if section in self.overrides:
                for (key, value) in self.overrides[section].items():
                    opts[key] = value

            setattr(self, section, opts)

    def info(self, logger=None):
        """Prints a summary of all options."""
        _print = logger.info if logger else print
        for section in self.sections:
            opts = getattr(self, section)
            _print("-" * (len(section) + 2))
            _print("[{}]".format(section))
            _print("-" * (len(section) + 2))
            for key, value in opts.items():
                if isinstance(value, list):
                    _print("{:>20}:".format(key))
                    for elem in value:
                        _print("{:>22}".format(elem))
                elif isinstance(value, dict):
                    _print("{:>20}:".format(key))
                    for k, v in value.items():
                        _print("{:>22}: {}".format(k, v))
                else:
                    _print("{:>20}: {}".format(key, value))
        _print("-" * 70)

    def to_dict(self):
        """Serializes the instance as dict."""
        d = {'filename': self.filename,
             'sections': self.sections}
        for section in self.sections:
            d[section] = copy.deepcopy(getattr(self, section))

        return d

    def __getitem__(self, key):
        return getattr(self, key)
