# -*- coding: utf-8 -*-
import math
from collections import defaultdict

import numpy as np

from torch.utils.data.sampler import Sampler


class BucketBatchSampler(Sampler):
    r"""Samples batch indices from sequence-length buckets efficiently
    with very little memory overhead.

    Epoch overhead for 5M dataset with batch_size=32 is around 400ms.

    Arguments:
        lengths (list): List of integer lengths corresponding to each
            item in the dataset.
        batch_size (int): Size of mini-batch.
        max_len (int, optional): A maximum sequence length that will be used
            to filter out very long sequences. A default of `10000` is
            assumed if ``None`` given.

    Example:
        # Generate dummy length information
        >> lengths = np.random.randint(1, 20, size=10000)
        >> sampler = BucketBatchSampler(lens, batch_size=10)
        >> batch = list(sampler)[0]
        >> batch
        [7526, 8473, 9194, 1030, 1568, 4182, 3082, 827, 3688, 9336]
        >> [lengths[i] for i in batch]
        # All samples in the batch have same length
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

    """

    def __init__(self, lengths, batch_size, max_len=None, store_indices=False):
        self.batch_size = batch_size
        self.max_len = 10000 if max_len is None else max_len
        self.store_indices = store_indices

        # Buckets: lengths -> list of sample indices
        self.buckets = defaultdict(list)

        # Fill the buckets while optionally filtering out long sequences
        for idx, len_ in enumerate(lengths):
            if len_ <= self.max_len:
                self.buckets[len_].append(idx)

        self.bucket_names = list(self.buckets.keys())

        # Pre-compute how many times a bucket will be sampled
        self.bucket_idxs = []

        for len_ in self.buckets:
            # Convery bucket to numpy array
            np_bucket = np.array(self.buckets[len_])

            # How many batches will be done for this bucket?
            bucket_bs = np_bucket.size / self.batch_size
            idxs = [len_] * math.ceil(bucket_bs)

            self.buckets[len_] = np_bucket
            self.bucket_idxs.extend(idxs)

        # Convert to numpy array
        self.bucket_idxs = np.array(self.bucket_idxs)

        # Set number of batches
        self.n_batches = len(self.bucket_idxs)

    def __iter__(self):
        # Keep offsets for each bucket for efficiency
        bucket_offsets = {}

        # Random access indices
        bucket_views = {}

        # If beam-search with ordered batches, original indices will be
        # necessary.
        self.orig_idxs = []

        # Create permuted access indices for each bucket
        # to avoid shuffling the lists
        for len_, elems in self.buckets.items():
            bucket_offsets[len_] = 0
            perms = np.random.permutation(len(elems))
            bucket_views[len_] = perms

        # Shuffle bucket order
        # For each bucket, slide the window to yield the next batch
        shuf_idxs = np.random.permutation(self.bucket_idxs)
        for bidx in shuf_idxs:
            # Get offset pointer for this bucket: 0 initially
            offset = bucket_offsets[bidx]

            # Convert them to permuted view
            idxs = bucket_views[bidx][offset: offset + self.batch_size]

            # Increment offset
            bucket_offsets[bidx] += len(idxs)

            # Get actual sample indices
            sidxs = self.buckets[bidx][idxs]

            if self.store_indices:
                self.orig_idxs.extend(sidxs)

            # Return sample indices
            yield sidxs

    def __len__(self):
        """Returns how many batches are inside."""
        return self.n_batches
