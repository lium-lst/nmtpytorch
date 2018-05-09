# -*- coding: utf-8 -*-
from torch.utils.data import Dataset

from ..utils.topology import DataSource


class MultiParallelDataset(Dataset):
    r"""A PyTorch dataset to fusion an arbitrary number of parallel data sources.

    Arguments:
        src_datasets(dict): dict of ``DataSource: torch.Dataset`` derived
            instances for source-side modalities.
        trg_datasets(dict, optional): dict of ``DataSource: torch.Dataset``
            derived instances for target-side modalities.
            Can be omitted for cases where no ground-truth targets exist.
    """

    def __init__(self, src_datasets, trg_datasets=None):
        self.sources = []
        self.targets = []
        self.data = {}

        if trg_datasets is None:
            trg_datasets = {}

        for key, dataset in src_datasets.items():
            assert isinstance(key, DataSource)
            self.sources.append(key)
            self.data[key] = dataset

        for key, dataset in trg_datasets.items():
            assert isinstance(key, DataSource)
            self.targets.append(key)
            self.data[key] = dataset

        assert not set(self.sources).intersection(self.targets), \
            "Same modality appearing on both source and target sides."

        self.sources = sorted(self.sources)
        self.targets = sorted(self.targets)
        self.n_sources = len(self.sources)
        self.n_targets = len(self.targets)

        # Detect dataset sizes
        sizes = set()
        for ds in self.data.values():
            sizes.add(len(ds))

        assert len(sizes) == 1, "Non-parallel datasets are not supported yet."

        # Set global size
        self.size = list(sizes)[0]

        # Set data sources to all by default
        self.set_data_sources()

    def set_data_sources(self, srcs=True, trgs=True):
        self.data_sources = []
        if isinstance(srcs, bool) and srcs:
            # Add all sources
            self.data_sources.extend(self.sources)

        if isinstance(trgs, bool) and trgs:
            self.data_sources.extend(self.targets)

    def __getitem__(self, idx):
        # We get a sample indice from the upstream sampler
        return {k: self.data[k][idx] for k in self.data_sources}

    def __len__(self):
        return self.size

    def __repr__(self):
        s = "{} - ({} sources / {} targets)\n".format(
            self.__class__.__name__, self.n_sources, self.n_targets)
        if self.n_sources > 0:
            s += "  Sources:"
            for name in self.sources:
                dstr = self.data[name].__repr__()
                s += '\n    ' + dstr.replace('\n', '\n    ') + '\n'
        if self.n_targets > 0:
            s += "  Targets:"
            for name in self.targets:
                dstr = self.data[name].__repr__()
                s += '\n    ' + dstr.replace('\n', '\n    ') + '\n'
        return s
