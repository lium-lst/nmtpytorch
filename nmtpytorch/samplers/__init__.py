import torch

from .bucket import BucketBatchSampler
from .approx import ApproximateBucketBatchSampler


def get_sampler(type_):
    return {
        'sequential': torch.utils.data.sampler.SequentialSampler,
        'random': torch.utils.data.sampler.RandomSampler,
        # length-aware
        'bucket': BucketBatchSampler,
        'approximate': ApproximateBucketBatchSampler,
    }[type_.lower()]
