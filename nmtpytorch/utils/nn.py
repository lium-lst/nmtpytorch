# -*- coding: utf-8 -*-
import numpy as np

from ..config import FLOAT


def freeze_parameters(module):
    """Disables updating the weights of CNN to use it as feature extractor."""
    for param in module.parameters():
        param.requires_grad = False


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


def get_network_topology(direction):
    """Parses a string in the form of 's1,s2,..,sN->t1,t2,..tN' and returns
    a dictionary representing encoder-decoder topology."""
    from ..utils.misc import LANGUAGES

    srcs, trgs = direction.split('->')
    srcs = srcs.split(',')
    trgs = trgs.split(',')
    src_languages = [s for s in srcs if s in LANGUAGES]
    trg_languages = [s for s in trgs if s in LANGUAGES]
    src_aux = list(set(srcs).difference(src_languages))
    trg_aux = list(set(trgs).difference(trg_languages))
    return {
        'multi_src': len(srcs) > 1,
        'multi_trg': len(trgs) > 1,
        'src_langs': src_languages,
        'trg_langs': trg_languages,
        'n_src_langs': len(src_languages),
        'n_trg_langs': len(trg_languages),
        'src_aux': src_aux,
        'trg_aux': trg_aux,
        'srcs': srcs,
        'trgs': trgs,
    }
