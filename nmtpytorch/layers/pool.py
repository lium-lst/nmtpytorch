import torch


class Pool(torch.nn.Module):
    """A pool layer with mean/max/sum/last options."""
    def __init__(self, op_type, pool_dim):
        super().__init__()

        self.op_type = op_type
        self.pool_dim = pool_dim
        assert self.op_type in ["last", "mean", "max", "sum"], \
            "Pool() operation should be mean, max, sum or last."

        if self.op_type == 'last':
            self.__pool_fn = lambda x: x.select(self.pool_dim, -1)
        elif self.op_type == 'max':
            pool_fn = getattr(torch, self.op_type)
            self.__pool_fn = lambda x: pool_fn(x, dim=self.pool_dim)[0]
        else:
            pool_fn = getattr(torch, self.op_type)
            self.__pool_fn = lambda x: pool_fn(x, dim=self.pool_dim)

    def forward(self, x):
        return self.__pool_fn(x)

    def __repr__(self):
        return "Pool(op_type={}, pool_dim={})".format(
            self.op_type, self.pool_dim)
