# -*- coding: utf-8 -*-
from .bucket import BucketBatchSampler
from .approx import ApproximateBucketBatchSampler

def get_sampler(type_):
    return {
        'bucket': BucketBatchSampler,
        'approximate': ApproximateBucketBatchSampler,
    }[type_.lower()]
