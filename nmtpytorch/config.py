# -*- coding: utf-8 -*-
import os
import copy
import pathlib

from collections import defaultdict

from configparser import ConfigParser, ExtendedInterpolation
from ast import literal_eval

# Default data types
INT = 'int64'
FLOAT = 'float32'

TRAIN_DEFAULTS = {
    'device_id': 'auto_1',       # auto_N for automatic N gpus
                                 # 0,1,2 for manual N gpus
                                 # 0 for 0th (single) GPU
    'seed': 1234,                # RNG seed. 0 -> Don't init
    'gclip': 5.,                 # Clip gradients above clip_c
    'l2_reg': 0.,                # L2 penalty factor
    'patience': 20,              # Early stopping patience
    'optimizer': 'adam',         # adadelta, sgd, rmsprop, adam
    'lr': 0.0004,                # 0 -> Use default lr from Pytorch
    'disp_freq': 30,             # Training display frequency (/batch)
    'batch_size': 32,            # Training batch size
    'max_epochs': 100,           # Max number of epochs to train
    'max_iterations': int(1e6),  # Max number of updates to train
    'eval_metrics': 'loss',      # comma sep. metrics, 1st -> earlystopping
    'eval_filters': '',          # comma sep. filters to apply to refs and hyps
    'eval_beam': 6,              # Validation beam size
    'eval_batch_size': 16,       # batch_size for GPU beam-search
    'eval_freq': 3000,           # 0 means 'End of epochs'
    'eval_start': 1,             # Epoch which validation will start
    'eval_zero': False,          # Evaluate once before starting training
                                 # Useful when using pretrained init.
    'save_best_metrics': True,   # Save best models for each eval_metric
    'checkpoint_freq': 5000,     # Periodic checkpoint frequency
    'n_checkpoints': 5,          # Number of checkpoints to keep
    'tensorboard_dir': '',       # Enable TB and give global log folder
}

def expand_env_vars(data):
    """Interpolate some environment variables."""
    data = data.replace('$HOME', os.environ['HOME'])
    data = data.replace('$USER', os.environ['USER'])
    return data


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
        self.__parser = ConfigParser(interpolation=ExtendedInterpolation())
        self.filename = filename
        self.overrides = defaultdict(dict)
        self.sections = []

        with open(self.filename) as f:
            data = expand_env_vars(f.read().strip())

        # Read the defaults first
        self.__parser.read_dict({'train': TRAIN_DEFAULTS})

        self.__parser.read_string(data)

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
