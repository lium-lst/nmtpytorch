import json
from abc import ABCMeta, abstractmethod


class BaseDataset(metaclass=ABCMeta):
    r"""Abstract dataset class that all nmtpytorch datasets should be
    extending.

    A sane enough `__repr__` is provided which will neatly dump the
    instance attributes not prefixed with `_`. So make sure to hide
    non-public attributes by prefixing them with `_` to avoid cluttering
    on the terminal. You can also redefine this method in the deriving
    classes.

    """
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """Returns the `idx`'th item of the dataset."""
        pass

    @abstractmethod
    def collate(self, elems):
        """Collates a list of elements into a `torch.Tensor`."""
        pass

    def get_batch_tensor(self, idxs):
        """Collates by looking up elements of the `idxs` list."""
        return self.collate([self.__getitem__(idx) for idx in idxs])

    def __repr__(self):
        public_attrs = [s for s in self.__dict__.keys() if s[0] != '_']
        info_dict = {k: self.__dict__[k] for k in public_attrs}
        public_str = json.dumps(
            info_dict, indent=1, default=lambda obj: str(obj))
        return f"\n{self.__class__.__name__} {public_str}\n"
