# -*- coding: utf-8 -*-
import sys
import copy
import json
import runpy
import pathlib

from difflib import get_close_matches

from collections import defaultdict


TRAIN_DEFAULTS = {
    'pretrained_file': '',       # A .ckpt file from which layers will be initialized
    'pretrained_layers': '',     # comma sep. list of layer prefixes to initialize
    'freeze_layers': '',         # comma sep. list of layer prefixes to freeze
    'handle_oom': False,         # Skip out-of-memory batches
    'seed': 0,                   # > 0 if you want to reproduce a previous experiment
    'save_optim_state': False,   # Save optimizer states into checkpoint
    #'gclip': 5.,                 # Clip gradients above clip_c
    #'l2_reg': 0.,                # L2 penalty factor
    #'patience': 20,              # Early stopping patience
    #'optimizer': 'adam',         # adadelta, sgd, rmsprop, adam
    #'lr': 0.0004,                # 0 -> Use default lr from Pytorch
    #'lr_decay': False,           # Can only be 'plateau' for now
    #'lr_decay_revert': False,    # Return back to the prev best weights after decay
    #'lr_decay_factor': 0.1,      # Check torch.optim.lr_scheduler
    #'lr_decay_patience': 10,     #
    #'lr_decay_min': 0.000001,    #
    #'momentum': 0.0,             # momentum for SGD
    #'nesterov': False,           # Enable Nesterov for SGD
    #'batch_size': 32,            # Training batch size
    #'max_epochs': 100,           # Max number of epochs to train
    #'max_iterations': int(1e6),  # Max number of updates to train
    #'eval_metrics': 'loss',      # comma sep. metrics, 1st -> earlystopping
    #'eval_filters': '',          # comma sep. filters to apply to refs/hyps
    #'eval_beam': 6,              # Validation beam size
    #'eval_batch_size': 16,       # batch_size for beam-search
    #'eval_freq': 3000,           # 0 means 'End of epochs'
    #'eval_max_len': 200,         # max seq len to stop during beam search
    #'eval_start': 1,             # Epoch which validation will start
    #'eval_zero': False,          # Evaluate once before starting training
                                 # Useful when using pretrained_file
    #'save_best_metrics': True,   # Save best models for each eval_metric
    #'checkpoint_freq': 5000,     # Periodic checkpoint frequency
    #'n_checkpoints': 5,          # Number of checkpoints to keep
}


def resolve_path(value):
    if isinstance(value, list):
        return [resolve_path(elem) for elem in value]
    if isinstance(value, dict):
        return {k: resolve_path(v) for k, v in value.items()}
    if isinstance(value, str) and value.startswith(('~', '/', '../', './')):
        return pathlib.Path(value).expanduser().resolve()
    return value


class Config:
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
            overrides[section][key] = value
        return overrides

    def __init__(self, filename, overrides=None):
        self.filename = filename
        self.sections = {
            'tasks': {},
            'train': TRAIN_DEFAULTS,
            'data': {},
            'vocabulary': {},
            'optimizers': {},
            'evaluation': {},
            'samplers': {},
        }

        if overrides is not None:
            # ex: train.batch_size:32
            self.overrides = self.parse_overrides(overrides)

        # Run the python file and get the relevant items
        namespace = runpy.run_path(filename)
        self.prepare_script = namespace.get('prepare', None)
        for section in self.sections:
            self.sections[section].update(namespace[section])
            self.sections[section].update(self.overrides[section])
            for key, value in self.sections[section].items():
                self.sections[section][key] = resolve_path(value)

        # Sanity check for `train`
        train_keys = list(self.sections['train'].keys())
        def_keys = list(TRAIN_DEFAULTS.keys())
        assert len(train_keys) == len(set(train_keys)), \
            "Duplicate arguments found in config's `train` section."

        invalid_keys = set(train_keys).difference(set(TRAIN_DEFAULTS))
        for key in invalid_keys:
            match = get_close_matches(key, def_keys, n=1)
            msg = "{}:train: Unknown option '{}'.".format(self.filename, key)
            if match:
                msg += "  Did you mean '{}' ?".format(match[0])
            print(msg)
        if invalid_keys:
            sys.exit(1)

        # Save the internal state for further serialization purposes
        self._state = self.__set_state()

    @property
    def state(self):
        return self._state

    def __repr__(self):
        return json.dumps(self.sections, indent=2, default=lambda obj: str(obj))

    def __set_state(self):
        """Serializes the instance as dict."""
        dict_ = {}
        for section, value in self.sections.items():
            dict_[section] = copy.deepcopy(value)
        self._state = dict_

    def __getattr__(self, key):
        return self.sections[key]
