# -*- coding: utf-8 -*-
import numpy as np

from ..config import FLOAT


def set_learnable(module, value=False, layer_names=''):
    """Disables updating the weights of CNN to use it as feature extractor."""
    # Can be empty or not
    layer_names = tuple(layer_names.split(','))

    for name, param in module.named_parameters():
        if any(layer_names):
            if name.startswith(layer_names):
                param.requires_grad = value
        else:
            param.requires_grad = value


def tile_ctx_dict(ctx_dict, idxs):
    """Returns dict of 3D tensors repeatedly indexed along the sample axis."""
    # 1st: tensor, 2nd optional mask
    return {
        k: (tensor[:, idxs], None if mask is None else mask[:, idxs])
        for k, (tensor, mask) in ctx_dict.items()
    }


def normalize_images(x):
    """Normalizes images in-place w.r.t ImageNet statistics."""
    mean = np.array([0.485, 0.456, 0.406], dtype=FLOAT)
    std = np.array([0.229, 0.224, 0.225], dtype=FLOAT)

    # Scale range
    x /= 255.

    # Normalize
    x -= mean[None, :, None, None]
    x /= std[None, :, None, None]
    return x
