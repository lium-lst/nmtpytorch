from torch.autograd import Variable


class Batch(object):
    def __init__(self, batch_size, data_dict):
        self.size = batch_size
        self.data = data_dict
        self.device = None

    def to_gpu(self, volatile=False):
        if self.device is None:
            self.device = 'gpu'
            self.data = {
                k: Variable(v, volatile=volatile).cuda() for k, v in self.data.items()}

    def to_cpu(self, volatile=False):
        if self.device is None:
            self.device = 'cpu'
            self.data = {
                k: Variable(v, volatile=volatile) for k, v in self.data.items()}

    def __getitem__(self, key):
        return self.data[key]

    def __repr__(self):
        s = "Batch(size={}, device={})\n".format(self.size, self.device)
        for key in self.data:
            s += "  {:10s} -> {}\n".format(str(key), self.data[key].shape)
        return s


def get_collate(data_sources):
    """Returns a special collate_fn which will view the underlying data
    in terms of the given DataSource keys."""

    def collate_fn(batch):
        return Batch(
            len(batch),
            {ds: ds.to_torch([elem[ds] for elem in batch]) for ds in data_sources},
        )

    return collate_fn
