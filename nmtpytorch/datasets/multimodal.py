# -*- coding: utf-8 -*-
import logging
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler

from . import get_dataset
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
            the batches will be bucketed, i.e. sort key. If `None`, no
            bucketing will be performed but the layers and models should
            support packing/padding/masking for this to work.
        max_len(int, optional): Maximum sequence length for ``bucket_by``
            modality to reject batches with long sequences. Does not have an effect
            if mode != 'train'.
        bucket_order (str, optional): ``ascending`` or ``descending`` to
            perform length-based curriculum learning. Default is ``None``
            which shuffles bucket order. Does not have an effect if mode != 'train'.
        sampler_type(str, optional): 'bucket' or 'approximate' (Default: 'bucket')
        kwargs (dict): Additional arguments to pass to the dataset constructors.
    """
    def __init__(self, data, mode, batch_size, vocabs, topology,
                 bucket_by, bucket_order=None, max_len=None,
                 sampler_type='bucket', **kwargs):
        self.datasets = {}
        self.mode = mode
        self.vocabs = vocabs
        self.batch_size = batch_size
        self.topology = topology
        self.bucket_by = bucket_by
        self.sampler_type = sampler_type

        # Disable filtering if not training
        self.max_len = max_len if self.mode == 'train' else None

        # This is only useful for training
        self.bucket_order = bucket_order if self.mode == 'train' else None

        # For old models to work, set it to the first source
        if self.bucket_by is None:
            logger.info(
                'WARNING: Bucketing sampler disabled. It is up to the model '
                'to take care of packing/padding/masking if any.')

        for key, ds in self.topology.all.items():
            if self.mode == 'beam' and ds.trg:
                # Skip target streams for beam-search
                continue

            try:
                # Get the relevant dataset class
                dataset_constructor = get_dataset(ds._type)
            except KeyError as ke:
                logger.info("ERROR: Unknown dataset type '{}'".format(ds._type))

            # Construct the dataset
            logger.info("Initializing dataset for '{}'".format(ds))
            self.datasets[ds] = dataset_constructor(
                fname=data[key],
                vocab=vocabs.get(key, None), bos=ds.trg, **kwargs)

        # Detect dataset sizes
        sizes = set([len(dataset) for dataset in self.datasets.values()])
        assert len(sizes) == 1, "Non-parallel datasets are not supported."

        # Set list of available datasets
        self.keys = list(self.datasets.keys())

        # Get collator
        self.collate_fn = get_collate(self.keys)

        if self.bucket_by is not None and self.bucket_by in self.datasets:
            if self.sampler_type == 'approximate':
                gen_sampler = ApproximateBucketBatchSampler
            elif self.sampler_type == 'bucket':
                gen_sampler = BucketBatchSampler
            self.sort_lens = self.datasets[self.bucket_by].lengths
            self.sampler = gen_sampler(
                batch_size=self.batch_size,
                sort_lens=self.sort_lens,
                max_len=self.max_len,
                store_indices=self.mode != 'train',
                order=self.bucket_order)
        else:
            # bucket_by only valid for training
            if self.bucket_by:
                self.bucket_by = None
                logger.info('Disabling bucketing for data loader.')
            # No modality provided to bucket sequential batches
            # Used for beam-search in image->text tasks
            if self.mode == 'beam':
                sampler = SequentialSampler(self)
                self.sampler_type = 'sequential'
            else:
                sampler = RandomSampler(self)
                self.sampler_type = 'random'
            self.sampler = BatchSampler(
                sampler, batch_size=self.batch_size, drop_last=False)

        # Set some metadata
        self.n_sources = len([k for k in self.keys if k.src])
        self.n_targets = len([k for k in self.keys if k.trg])

        # Set dataset size
        self.size = list(sizes)[0]

    def __getitem__(self, idx):
        return {k: self.datasets[k][idx] for k in self.keys}

    def __len__(self):
        return self.size

    def __repr__(self):
        s = "{} - ({} source(s) / {} target(s))\n".format(
            self.__class__.__name__, self.n_sources, self.n_targets)
        s += "  Sampler type: {}, bucket_by: {}\n".format(
            self.sampler_type, self.bucket_by)

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
