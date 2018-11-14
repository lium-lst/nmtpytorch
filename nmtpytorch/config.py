# -*- coding: utf-8 -*-
import os
import sys
import copy
import pathlib
from difflib import get_close_matches

from collections import defaultdict

from configparser import ConfigParser, ExtendedInterpolation
from ast import literal_eval


TRAIN_DEFAULTS = {
    'num_workers': 0,            # number of workers for data loading (0=disabled)
    'pin_memory': False,         # pin_memory for DataLoader (Default: False)
    'seed': 0,                   # > 0 if you want to reproduce a previous experiment
    'gclip': 5.,                 # Clip gradients above clip_c
    'l2_reg': 0.,                # L2 penalty factor
    'patience': 20,              # Early stopping patience
    'optimizer': 'adam',         # adadelta, sgd, rmsprop, adam
    'lr': 0.0004,                # 0 -> Use default lr from Pytorch
    'lr_decay': False,           # Can only be 'plateau' for now
    'lr_decay_revert': False,    # Return back to the prev best weights after decay
    'lr_decay_factor': 0.1,      # Check torch.optim.lr_scheduler
    'lr_decay_patience': 10,     #
    'lr_decay_min': 0.000001,    #
    'model_type': '',            # Name of model class to train
    'momentum': 0.0,             # momentum for SGD
    'nesterov': False,           # Enable Nesterov for SGD
    'disp_freq': 30,             # Training display frequency (/batch)
    'batch_size': 32,            # Training batch size
    'max_epochs': 100,           # Max number of epochs to train
    'max_iterations': int(1e6),  # Max number of updates to train
    'eval_metrics': 'loss',      # comma sep. metrics, 1st -> earlystopping
    'eval_filters': '',          # comma sep. filters to apply to refs/hyps
    'eval_beam': 6,              # Validation beam size
    'eval_batch_size': 16,       # batch_size for beam-search
    'eval_freq': 3000,           # 0 means 'End of epochs'
    'eval_max_len': 200,         # max seq len to stop during beam search
    'eval_start': 1,             # Epoch which validation will start
    'eval_zero': False,          # Evaluate once before starting training
                                 # Useful when using pretrained_file
    'save_best_metrics': True,   # Save best models for each eval_metric
    'save_path': '',             # Path to root experiment folder
    'checkpoint_freq': 5000,     # Periodic checkpoint frequency
    'n_checkpoints': 5,          # Number of checkpoints to keep
    'tensorboard_dir': '',       # Enable TB and give global log folder
    'pretrained_file': '',       # A .ckpt file from which layers will be initialized
    'freeze_layers': '',         # comma sep. list of layer prefixes to freeze
    'handle_oom': False,         # Skip out-of-memory batches
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
    if isinstance(value, dict):
        return {k: resolve_path(v) for k, v in value.items()}
    if isinstance(value, str) and value.startswith(('~', '/', '../', './')):
        return pathlib.Path(value).expanduser().resolve()
    return value


def _parse_value(value):
    """Automatic type conversion for configuration values.

    Arguments:
        value(str): A string to parse.
    """

    # Check for boolean or None
    if str(value).capitalize().startswith(('False', 'True', 'None')):
        return eval(str(value).capitalize(), {}, {})

    # Detect strings, floats and ints
    try:
        # If this fails, this is a string
        result = literal_eval(value)
    except Exception:
        result = value

    return result


class Options:
    @classmethod
    def from_dict(cls, dict_, override_list=None):
        """Loads object from dict."""
        obj = cls.__new__(cls)
        obj.__dict__.update(dict_)

        # Test time overrides are possible as well
        if override_list is not None:
            overrides = obj.parse_overrides(override_list)
            for section, ov_dict in overrides.items():
                for key, value in ov_dict.items():
                    obj.__dict__[section][key] = value

        return obj

    @classmethod
    def parse_overrides(cls, override_list):
        overrides = defaultdict(dict)
        for opt in override_list:
            section, keyvalue = opt.split('.', 1)
            key, value = keyvalue.split(':')
            value = resolve_path(value)
            overrides[section][key] = _parse_value(value)
        return overrides

    def __init__(self, filename, overrides=None):
        self._parser = ConfigParser(interpolation=ExtendedInterpolation())
        self.filename = filename
        self.sections = []

        with open(self.filename) as fhandle:
            data = expand_env_vars(fhandle.read().strip())

        # Read the defaults first
        self._parser.read_dict({'train': TRAIN_DEFAULTS})

        # Read the config
        self._parser.read_string(data)

        if overrides is not None:
            # ex: train.batch_size:32
            self.overrides = self.parse_overrides(overrides)

        for section in self._parser.sections():
            opts = {}
            self.sections.append(section)

            for key, value in self._parser[section].items():
                opts[key] = resolve_path(_parse_value(value))

            if section in self.overrides:
                for (key, value) in self.overrides[section].items():
                    opts[key] = value

            setattr(self, section, opts)

        # Sanity check for [train]
        train_keys = list(self.train.keys())
        def_keys = list(TRAIN_DEFAULTS.keys())
        assert len(train_keys) == len(set(train_keys)), \
            "Duplicate arguments found in config's [train] section."

        invalid_keys = set(train_keys).difference(set(TRAIN_DEFAULTS))
        for key in invalid_keys:
            match = get_close_matches(key, def_keys, n=1)
            msg = "{}:train: Unknown option '{}'.".format(self.filename, key)
            if match:
                msg += "  Did you mean '{}' ?".format(match[0])
            print(msg)
        if invalid_keys:
            sys.exit(1)

    def __repr__(self):
        repr_ = ""
        for section in self.sections:
            opts = getattr(self, section)
            repr_ += "-" * (len(section) + 2)
            repr_ += "\n[{}]\n".format(section)
            repr_ += "-" * (len(section) + 2)
            repr_ += '\n'
            for key, value in opts.items():
                if isinstance(value, list):
                    repr_ += "{:>20}:\n".format(key)
                    for elem in value:
                        repr_ += "{:>22}\n".format(elem)
                elif isinstance(value, dict):
                    repr_ += "{:>20}:\n".format(key)
                    for kkey, vvalue in value.items():
                        repr_ += "{:>22}:{}\n".format(kkey, vvalue)
                else:
                    repr_ += "{:>20}:{}\n".format(key, value)
        repr_ += "-" * 70
        repr_ += "\n"
        return repr_

    def to_dict(self):
        """Serializes the instance as dict."""
        dict_ = {
            'filename': self.filename,
            'sections': self.sections,
        }
        for section in self.sections:
            dict_[section] = copy.deepcopy(getattr(self, section))

        return dict_

    def __getitem__(self, key):
        return getattr(self, key)
