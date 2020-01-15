class Batch:
    """A custom batch object."""
    def __init__(self, data, device=None):
        if device:
            self.data = {k: v.to(device) for k, v in data.items()}
        else:
            self.data = data

        dim1s = set([x.size(1) for x in data.values()])
        assert len(dim1s) == 1, \
            "Incompatible batch dimension (1) between modalities."
        self.size = dim1s.pop()

    def __getitem__(self, key):
        return self.data[key]

    def to(self, device):
        self.data.update({k: v.to(device) for k, v in self.data.items()})

    def __repr__(self):
        s = f"Batch(size={self.size})\n"
        for key, value in self.data.items():
            s += f"  {key:10s} -> {value.shape} - {value.device}\n"
        return s[:-1]
