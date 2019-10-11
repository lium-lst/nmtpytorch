import torch


class Pool(torch.nn.Module):
    """A pool layer with mean/max/sum/last options."""
    def __init__(self, op_type, pool_dim, keepdim=True):
        super().__init__()

        self.op_type = op_type
        self.pool_dim = pool_dim
        self.keepdim = keepdim
        assert self.op_type in ["last", "mean", "max", "sum"], \
            "Pool() operation should be mean, max, sum or last."

        if self.op_type == 'last':
            self.__pool_fn = lambda x: x.select(
                self.pool_dim, -1).unsqueeze(0)
        else:
            if self.op_type == 'max':
                self.__pool_fn = lambda x: torch.max(
                    x, dim=self.pool_dim, keepdim=self.keepdim)[0]
            elif self.op_type == 'mean':
                self.__pool_fn = lambda x: torch.mean(
                    x, dim=self.pool_dim, keepdim=self.keepdim)
            elif self.op_type == 'sum':
                self.__pool_fn = lambda x: torch.sum(
                    x, dim=self.pool_dim, keepdim=self.keepdim)

    def forward(self, x):
        return self.__pool_fn(x)

    def __repr__(self):
        return "Pool(op_type={}, pool_dim={}, keepdim={})".format(
            self.op_type, self.pool_dim, self.keepdim)
