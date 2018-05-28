# -*- coding: utf-8 -*-
import torch.optim
from torch.nn.utils.clip_grad import clip_grad_norm

# Setup optimizer (should always come after model.cuda())
# iterable of dicts for per-param options where each dict
# is {'params' : [p1, p2, p3...]}.update(generic optimizer args)
# Example:
# optim.SGD([
        # {'params': model.base.parameters()},
        # {'params': model.classifier.parameters(), 'lr': 1e-3}
    # ], lr=1e-2, momentum=0.9)


class Optimizer(object):
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
                 nesterov=False, weight_decay=0, gclip=0):
        self.name = name
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.gclip = gclip

        self.optim_args = {}
        # If an explicit lr given, pass it to torch optimizer
        if self.lr > 0:
            self.optim_args['lr'] = self.lr

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
        self.lr = self.optim.defaults['lr']

        # Assign shortcuts
        self.zero_grad = self.optim.zero_grad

        if self.gclip == 0:
            self.step = self.optim.step
        else:
            self.step = self._step

    def _step(self, closure=None):
        """Gradient clipping aware step()."""
        clip_grad_norm(self.params, self.gclip)
        self.optim.step(closure)

    def __repr__(self):
        s = "Optimizer => {} (lr: {}, weight_decay: {}, g_clip: {}".format(
            self.name, self.lr, self.weight_decay, self.gclip)
        if self.name == 'sgd':
            s += ', momentum: {}, nesterov: {}'.format(
                self.momentum, self.nesterov)
        s += ')'
        return s
