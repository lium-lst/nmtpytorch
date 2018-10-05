# -*- coding: utf-8 -*-
import math
import logging
from collections import defaultdict

import numpy as np

from ..utils.device import DEVICE_IDS
from . import BucketBatchSampler

logger = logging.getLogger('nmtpytorch')


class ApproximateBucketBatchSampler(BucketBatchSampler):
    r"""Samples batch indices from sequence-length buckets efficiently
    with very little memory overhead.

    Different from `BucketBatchSampler`, this class bins data samples w.r.t
    lengths but does not guarantee that each bucket contains necessarily
    same-length sequences. Further padding/packing/masking should be done
    by detecting possible <pad> items in tensors.

    Arguments:
        batch_size (int): Size of mini-batch.
        sort_lens (list): List of source or target lengths corresponding to each
            item in the dataset.
        max_len (int, optional): A maximum sequence length that will be used
            to filter out very long sequences. ``None`` means no filtering.
        store_indices (bool, optional): If ``True``, indices that will unsort
            the dataset will be stored. This used by beam search/inference.
        order (str, optional): Default is ``None``, i.e. buckets are shuffled.
            If ``ascending`` or ``descending``, will iterate w.r.t bucket
            lengths to implement length-based curriculum learning.
    """

    def __init__(self, batch_size, sort_lens, max_len=None,
                 store_indices=False, order=None):
        assert order in (None, 'ascending', 'descending'), \
            "order should be None, 'ascending' or 'descending'"

        self.batch_size = batch_size
        self.max_len = max_len
        self.n_rejects = 0
        self.order = order
        self.store_indices = store_indices

        # Additional balancing logic for multi-GPU
        self.n_devices = len(DEVICE_IDS) if DEVICE_IDS else 1

        # Buckets: sort_lens -> list of sample indices
        self.buckets = defaultdict(list)

        # Pre-compute how many times a bucket will be sampled
        self.bucket_idxs = []

        # Fill the buckets while optionally filtering out long sequences
        if self.max_len is not None:
            for idx, len_ in enumerate(sort_lens):
                if len_ <= self.max_len:
                    self.buckets[len_].append(idx)
                else:
                    self.n_rejects += 1
            logger.info('{} samples rejected because of length filtering @ {}'.format(
                self.n_rejects, self.max_len))
        else:
            # No length filtering
            for idx, len_ in enumerate(sort_lens):
                self.buckets[len_].append(idx)

        ######################################
        # Modified part compared to base class
        ######################################
        ordered_idxs = []
        min_bucket_size = self.batch_size * 5
        for length in sorted(self.buckets):
            ordered_idxs.extend(self.buckets[length])

        # Reset buckets
        self.buckets = {}
        n_elems = len(ordered_idxs)

        # Bin sorted buckets approximately
        for idx, start in enumerate(range(0, n_elems, min_bucket_size)):
            self.buckets[idx] = ordered_idxs[start:start + min_bucket_size]

        # number of elems in the last bucket
        last_bucket_size = len(self.buckets[idx])
        # number of elems in the last batch of last bucket
        last_batch_size = last_bucket_size % self.batch_size
        # how many should we remove to make the last batch divisible for
        # many GPUs
        n_remove_from_last = last_batch_size % self.n_devices
        end_point = last_bucket_size - n_remove_from_last
        self.buckets[idx] = self.buckets[idx][:end_point]
        if n_remove_from_last > 0:
            logger.info('Removed {} samples to balance buckets.'.format(
                n_remove_from_last))

        self.stats = {k: len(self.buckets[k]) for k in sorted(self.buckets)}

        for len_ in self.buckets:
            # Convert bucket to numpy array
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
