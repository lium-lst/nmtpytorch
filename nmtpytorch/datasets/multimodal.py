# -*- coding: utf-8 -*-
import logging
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, SequentialSampler

from . import ImageFolderDataset, TextDataset, OneHotDataset, NumpyDataset
from .collate import get_collate
from ..samplers import BucketBatchSampler

logger = logging.getLogger('nmtpytorch')


class MultimodalDataset(Dataset):
    """Returns a Dataset for parallel multimodal corpus.

    Arguments:
        data_dict(dict): [data] section's relevant split dictionary
        vocabs(dict): dictionary mapping lang keys to Vocabulary() objects
        topology(Topology): A topology object.
        bucket_by(str): String identifier of the modality which will define how
            the batches will be bucketed, i.e. sort key.
        kwargs (dict): Argument dictionary for the ImageFolder dataset.
    """
    def __init__(self, data_dict, vocabs, topology, bucket_by, **kwargs):
        self.datasets = {}
        self.vocabs = vocabs
        self.topology = topology

        # For old models to work, set it to the first source
        if bucket_by is None:
            bucket_by = self.topology.get_src_langs()[0]

        for key, ds in self.topology.all.items():
            if key == bucket_by:
                self.bucket_by = ds

            if ds._type == "Text":
                # Prepend <bos> if datasource is on target side
                self.datasets[ds] = TextDataset(data_dict[key], vocabs[key],
                                                bos=ds.trg)
            elif ds._type == "OneHot":
                self.datasets[ds] = OneHotDataset(data_dict[key], vocabs[key])
            elif ds._type == "ImageFolder":
                self.datasets[ds] = ImageFolderDataset(data_dict[key], **kwargs)
            elif ds._type == "Numpy":
                self.datasets[ds] = NumpyDataset(data_dict[key])

        # Detect dataset sizes
        sizes = set()
        for dataset in self.datasets.values():
            sizes.add(len(dataset))
        assert len(sizes) == 1, "Non-parallel datasets are not supported."

        # Set dataset size
        self.size = list(sizes)[0]

        # Set list of available datasets
        self.keys = list(self.datasets.keys())

        self.n_sources = len([k for k in self.keys if k.src])
        self.n_targets = len([k for k in self.keys if k.trg])

    def get_loader_args(self, batch_size, drop_targets=False, inference=False):
        """Returns a BatchSampler instance."""
        if drop_targets and self.bucket_by.trg:
            sampler = BatchSampler(
                SequentialSampler(self),
                batch_size=batch_size, drop_last=False)
        else:
            sampler = BucketBatchSampler(
                batch_size=batch_size,
                sort_lens=self.datasets[self.bucket_by].lengths,
                store_indices=inference)

        # Reject target datasets if requested
        keys = list(filter(lambda k: not drop_targets or k.src, self.keys))

        return {
            'batch_sampler': sampler,
            'collate_fn': get_collate(keys),
        }

    def __getitem__(self, idx):
        return {k: self.datasets[k][idx] for k in self.keys}

    def __len__(self):
        return self.size

    def __repr__(self):
        s = "{} - ({} sources / {} targets)\n".format(
            self.__class__.__name__, self.n_sources, self.n_targets)
        s += "Batches sorted by '{}' lengths\n".format(self.bucket_by)
        if self.n_sources > 0:
            s += "  Sources:\n"
            for name in filter(lambda k: k.src, self.keys):
                dstr = self.datasets[name].__repr__()
                s += '    --> ' + dstr
        if self.n_targets > 0:
            s += "  Targets:\n"
            for name in filter(lambda k: k.trg, self.keys):
                dstr = self.datasets[name].__repr__()
                s += '    --> ' + dstr
        return s
