# -*- coding: utf-8 -*-
import logging

import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_

logger = logging.getLogger('nmtpytorch')

# Setup optimizer (should always come after model.to())
# iterable of dicts for per-param options where each dict
# is {'params' : [p1, p2, p3...]}.update(generic optimizer args)
# Example:
# optim.SGD([
        # {'params': model.base.parameters()},
        # {'params': model.classifier.parameters(), 'lr': 1e-3}
    # ], lr=1e-2, momentum=0.9)


class Optimizer:
    # Class dict to map lowercase identifiers to actual classes
    methods = {
        'adadelta':   torch.optim.Adadelta,
        'adagrad':    torch.optim.Adagrad,
        'adam':       torch.optim.Adam,
        'sgd':        torch.optim.SGD,
        'asgd':       torch.optim.ASGD,
        'rprop':      torch.optim.Rprop,
        'rmsprop':    torch.optim.RMSprop,
    }

    @staticmethod
    def get_params(model):
        """Returns all name, parameter pairs with requires_grad=True."""
        return list(
            filter(lambda p: p[1].requires_grad, model.named_parameters()))

    def __init__(self, name, model, lr=0, momentum=0.0,
                 nesterov=False, weight_decay=0, gclip=0,
                 lr_decay=False, lr_decay_factor=0.1, lr_decay_mode='min',
                 lr_decay_patience=10, lr_decay_min=0.000001):
        self.name = name
        self.model = model
        self.initial_lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_mode = lr_decay_mode
        self.lr_decay_patience = lr_decay_patience
        self.lr_decay_min = lr_decay_min
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.gclip = gclip

        self.optim_args = {}
        # If an explicit lr given, pass it to torch optimizer
        if self.initial_lr > 0:
            self.optim_args['lr'] = self.initial_lr

        if self.name == 'sgd':
            self.optim_args['momentum'] = self.momentum
            self.optim_args['nesterov'] = self.nesterov

        # Get all parameters that require grads
        self.named_params = self.get_params(self.model)

        # Filter out names for gradient clipping
        self.params = [param for (name, param) in self.named_params]

        if self.weight_decay > 0:
            weight_group = {
                'params': [p for n, p in self.named_params if 'bias' not in n],
                'weight_decay': self.weight_decay,
            }
            bias_group = {
                'params': [p for n, p in self.named_params if 'bias' in n],
            }
            self.param_groups = [weight_group, bias_group]

        else:
            self.param_groups = [{'params': self.params}]

        # Safety check
        n_params = len(self.params)
        for group in self.param_groups:
            n_params -= len(group['params'])
        assert n_params == 0, "Not all params are passed to the optimizer."

        # Create the actual optimizer
        self.optim = self.methods[self.name](self.param_groups, **self.optim_args)

        # Get final lr that will be used
        self.initial_lr = self.optim.defaults['lr']
        self.cur_lr = self.initial_lr

        # Assign shortcuts
        self.zero_grad = self.optim.zero_grad

        if self.gclip == 0:
            self.step = self.optim.step
        else:
            self.step = self._step

        if self.lr_decay:
            self.scheduler = ReduceLROnPlateau(
                self.optim, mode=self.lr_decay_mode,
                factor=self.lr_decay_factor, patience=self.lr_decay_patience,
                min_lr=self.lr_decay_min)
        else:
            self.scheduler = None

    def _step(self, closure=None):
        """Gradient clipping aware step()."""
        clip_grad_norm_(self.params, self.gclip)
        self.optim.step(closure)

    def lr_step(self, metric):
        if self.scheduler is not None:
            self.scheduler.step(metric)
            if self.get_lr() != self.cur_lr:
                self.cur_lr = self.get_lr()
                logger.info('** Learning rate changed -> {}'.format(self.cur_lr))
                # Signal it back
                return True
        return False

    def get_lr(self):
        """Returns current lr for parameters."""
        return self.optim.param_groups[0]['lr']

    def __repr__(self):
        repr_ = "Optimizer => {} (lr: {}, weight_decay: {}, g_clip: {}".format(
            self.name, self.initial_lr, self.weight_decay, self.gclip)
        if self.name == 'sgd':
            repr_ += ', momentum: {}, nesterov: {}'.format(
                self.momentum, self.nesterov)
        if self.lr_decay:
            repr_ += ', lr_decay: (patience={}, factor={})'.format(
                self.lr_decay_patience, self.lr_decay_factor)
        repr_ += ')'
        return repr_
