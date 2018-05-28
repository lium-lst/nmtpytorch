# -*- coding: utf-8 -*-
import os
import copy
import pathlib

from collections import defaultdict

from configparser import ConfigParser, ExtendedInterpolation
from ast import literal_eval


TRAIN_DEFAULTS = {
    'device_id': 'auto_1',       # auto_N for automatic N gpus
                                 # 0,1,2 for manual N gpus
                                 # 0 for 0th (single) GPU
    'num_workers': 0,            # number of workers for data loading (0=disabled)
    'pin_memory': False,         # pin_memory for DataLoader (Default: False)
    'seed': 0,                   # > 0 if you want to reproduce a previous experiment
    'gclip': 5.,                 # Clip gradients above clip_c
    'l2_reg': 0.,                # L2 penalty factor
    'patience': 20,              # Early stopping patience
    'optimizer': 'adam',         # adadelta, sgd, rmsprop, adam
    'lr': 0.0004,                # 0 -> Use default lr from Pytorch
    'momentum': 0.0,             # momentum for SGD
    'nesterov': False,           # Enable Nesterov for SGD
    'disp_freq': 30,             # Training display frequency (/batch)
    'batch_size': 32,            # Training batch size
    'max_epochs': 100,           # Max number of epochs to train
    'max_iterations': int(1e6),  # Max number of updates to train
    'eval_metrics': 'loss',      # comma sep. metrics, 1st -> earlystopping
    'eval_filters': '',          # comma sep. filters to apply to refs/hyps
    'eval_beam': 6,              # Validation beam size
    'eval_batch_size': 16,       # batch_size for GPU beam-search
    'eval_freq': 3000,           # 0 means 'End of epochs'
    'eval_start': 1,             # Epoch which validation will start
    'eval_zero': False,          # Evaluate once before starting training
                                 # Useful when using pretrained_file
    'save_best_metrics': True,   # Save best models for each eval_metric
    'checkpoint_freq': 5000,     # Periodic checkpoint frequency
    'n_checkpoints': 5,          # Number of checkpoints to keep
    'tensorboard_dir': '',       # Enable TB and give global log folder
    'pretrained_file': '',       # A .ckpt file from which layers will be initialized
    'freeze_layers': '',         # comma sep. list of layer prefixes to freeze
}


def expand_env_vars(data):
    """Interpolate some environment variables."""
    for key in ('HOME', 'USER', 'LOCAL', 'SCRATCH'):
        var = '$' + key
        if var in data and key in os.environ:
            data = data.replace(var, os.environ[key])
    return data


def resolve_path(value):
    if isinstance(value, list):
        return [resolve_path(elem) for elem in value]
    elif isinstance(value, dict):
        return {k: resolve_path(v) for k, v in value.items()}
    elif isinstance(value, str) and value.startswith(('~', '/', '../', './')):
        return pathlib.Path(value).expanduser().resolve()
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

    def __repr__(self):
        s = ""
        for section in self.sections:
            opts = getattr(self, section)
            s += "-" * (len(section) + 2)
            s += "\n[{}]\n".format(section)
            s += "-" * (len(section) + 2)
            s += '\n'
            for key, value in opts.items():
                if isinstance(value, list):
                    s += "{:>20}:\n".format(key)
                    for elem in value:
                        s += "{:>22}\n".format(elem)
                elif isinstance(value, dict):
                    s += "{:>20}:\n".format(key)
                    for k, v in value.items():
                        s += "{:>22}:{}\n".format(k, v)
                else:
                    s += "{:>20}:{}\n".format(key, value)
        s += "-" * 70
        s += "\n"
        return s

    def to_dict(self):
        """Serializes the instance as dict."""
        d = {'filename': self.filename,
             'sections': self.sections}
        for section in self.sections:
            d[section] = copy.deepcopy(getattr(self, section))

        return d

    def __getitem__(self, key):
        return getattr(self, key)
