# -*- coding: utf-8 -*-
import logging
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, SequentialSampler

from . import (ImageFolderDataset, TextDataset, OneHotDataset, NumpyDataset,
               KaldiDataset, NumpySequenceDataset, ShelveDataset)
from .collate import get_collate
from ..samplers import BucketBatchSampler, ApproximateBucketBatchSampler

logger = logging.getLogger('nmtpytorch')


class MultimodalDataset(Dataset):
    """Returns a Dataset for parallel multimodal corpora

    Arguments:
        data(dict): [data] section's relevant split dictionary
        mode(str): One of train/eval/beam.
        batch_size(int): Batch size.
        vocabs(dict): dictionary mapping keys to Vocabulary() objects
        topology(Topology): A topology object.
        bucket_by(str): String identifier of the modality which will define how
            the batches will be bucketed, i.e. sort key.
        max_len(int, optional): Maximum sequence length for ``bucket_by``
            modality to reject batches with long sequences.
        bucket_order (str, optional): ``ascending`` or ``descending`` to
            perform length-based curriculum learning. Default is ``None``
            which shuffles bucket order.
        sampler_type(str, optional): 'bucket' or 'approximate' (Default: 'bucket')
        kwargs (dict): Argument dictionary for datasets.
    """
    def __init__(self, data, mode, batch_size, vocabs, topology,
                 bucket_by, max_len=None, bucket_order=None,
                 sampler_type='bucket', **kwargs):
        self.datasets = {}
        self.mode = mode
        self.vocabs = vocabs
        self.batch_size = batch_size
        self.topology = topology
        self.bucket_by = bucket_by
        if sampler_type == 'approximate':
            gen_sampler = ApproximateBucketBatchSampler
        elif sampler_type == 'bucket':
            gen_sampler = BucketBatchSampler

        # Disable filtering if not training
        self.max_len = max_len if self.mode == 'train' else None
        self.bucket_order = bucket_order if self.mode == 'train' else None

        # For old models to work, set it to the first source
        if self.bucket_by is None:
            if len(self.topology.get_src_langs()) > 0:
                self.bucket_by = self.topology.get_src_langs()[0]
            elif self.mode != 'beam' and len(self.topology.get_trg_langs()) > 0:
                self.bucket_by = self.topology.get_trg_langs()[0]

        # TODO: This should be agnostic to datasets, i.e. no explicit calls
        # to dataset constructors should be necessary.
        for key, ds in self.topology.all.items():
            if self.mode == 'beam' and ds.trg:
                # Skip target streams
                continue

            if key == self.bucket_by:
                self.bucket_by = ds

            if ds._type == "Text":
                # Prepend <bos> if datasource is on target side
                self.datasets[ds] = TextDataset(
                    data[key], vocabs[key], bos=ds.trg)
            elif ds._type == "OneHot":
                self.datasets[ds] = OneHotDataset(data[key], vocabs[key])
            elif ds._type == "ImageFolder":
                self.datasets[ds] = ImageFolderDataset(data[key], **kwargs)
            elif ds._type == "Numpy":
                self.datasets[ds] = NumpyDataset(data[key])
            elif ds._type == "Shelve":
                self.datasets[ds] = ShelveDataset(data[key], **kwargs)
            elif ds._type == "Kaldi":
                self.datasets[ds] = KaldiDataset(data[key])
            elif ds._type == "NumpySequence":
                self.datasets[ds] = NumpySequenceDataset(data[key], **kwargs)
            else:
                raise ValueError("Unknown dataset type: {}.".format(ds))

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

        self.collate_fn = get_collate(self.keys)
        if self.bucket_by is not None:
            self.sort_lens = self.datasets[self.bucket_by].lengths
            self.sampler = gen_sampler(
                batch_size=self.batch_size,
                sort_lens=self.sort_lens,
                max_len=self.max_len,
                store_indices=self.mode == 'beam',
                order=self.bucket_order)
        else:
            # No modality to sort batches, return sequential data
            # Used for beam-search in image->text tasks
            self.sampler = BatchSampler(
                SequentialSampler(self),
                batch_size=self.batch_size, drop_last=False)

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
