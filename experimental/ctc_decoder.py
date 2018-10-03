# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from . import FF


class CTCDecoder(nn.Module):
    """A CTC decoder for speech."""
    def __init__(self, input_size, n_vocab, ctx_name):
        super().__init__()

        # Other arguments
        self.input_size = input_size
        self.n_vocab = n_vocab
        self.ctx_name = ctx_name

        # Project encodings to output vocabulary
        # +1 for the blank symbol
        self.hid2prob = FF(self.input_size, self.n_vocab + 1)

        try:
            from warpctc_pytorch import CTCLoss
        except ImportError as ie:
            raise RuntimeError('warpctc_pytorch is not installed.')

        # Let's not average for now w.r.t frame lengths
        self.loss = CTCLoss(size_average=False)

    def forward(self, ctx_dict, y=None):
        ctx, mask = ctx_dict[self.ctx_name]

        # Project to token vocabulary
        out = self.hid2prob(ctx)

        if y is not None:
            # Convert to CPU tensors
            y_cpu_int = y.cpu().int()

            # Do not count <pad>'s which are now 1
            label_lens = (y_cpu_int != 1).int().sum(0)

            labels = torch.masked_select(y_cpu_int, (y_cpu_int != 1)).view(-1)

            # Length of encoder states
            act_lens = Variable(
                torch.ones(out.shape[1]).int() * out.shape[0])

            loss = self.loss(out, labels, act_lens, label_lens)
        else:
            loss = None

        # For test scores
        logps = F.logsoftmax(out)
        return {'loss': loss, 'logps': logps}
