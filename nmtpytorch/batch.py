# -*- coding: utf-8 -*-
from torch.autograd import Variable


class Batch(object):
    def __init__(self, data_dict, batch_size):
        self.data = data_dict
        self.keys = list(self.data.keys())
        self.batch_size = batch_size
        self.device = "N/A"

    def to_gpu(self, volatile=False):
        self.device = 'gpu'
        self.data = {
            k: Variable(v, volatile=volatile).cuda() for k, v in self.data.items()}

    def to_cpu(self, volatile=False):
        self.device = 'cpu'
        self.data = {
            k: Variable(v, volatile=volatile) for k, v in self.data.items()}

    def __repr__(self):
        s = "Batch(size={}, device={})\n".format(self.batch_size, self.device)
        for key in self.keys:
            s += "  {:10s} -> {}\n".format(str(key), self.data[key].shape)
        return s
