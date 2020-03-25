# -*- coding: utf-8 -*-
import os
import bz2
import gzip
import lzma
import time
import random
import pathlib
import logging
import tempfile
from hashlib import sha256

import numpy as np
import torch
from tqdm import tqdm

from ..cleanup import cleanup

logger = logging.getLogger('nmtpytorch')


LANGUAGES = [
    'aa', 'ab', 'ae', 'af', 'ak', 'am', 'an', 'ar', 'as', 'av', 'ay', 'az',
    'ba', 'be', 'bg', 'bh', 'bi', 'bm', 'bn', 'bo', 'br', 'bs', 'ca', 'ce',
    'ch', 'co', 'cr', 'cs', 'cu', 'cv', 'cy', 'da', 'de', 'dv', 'dz', 'ee',
    'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'ff', 'fi', 'fj', 'fo', 'fr',
    'fy', 'ga', 'gd', 'gl', 'gn', 'gu', 'gv', 'ha', 'he', 'hi', 'ho', 'hr',
    'ht', 'hu', 'hy', 'hz', 'ia', 'id', 'ie', 'ig', 'ii', 'ik', 'io', 'is',
    'it', 'iu', 'ja', 'jv', 'ka', 'kg', 'ki', 'kj', 'kk', 'kl', 'km', 'kn',
    'ko', 'kr', 'ks', 'ku', 'kv', 'kw', 'ky', 'la', 'lb', 'lg', 'li', 'ln',
    'lo', 'lt', 'lu', 'lv', 'mg', 'mh', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms',
    'mt', 'my', 'na', 'nb', 'nd', 'ne', 'ng', 'nl', 'nn', 'no', 'nr', 'nv',
    'ny', 'oc', 'oj', 'om', 'or', 'os', 'pa', 'pi', 'pl', 'ps', 'pt', 'qu',
    'rm', 'rn', 'ro', 'ru', 'rw', 'sa', 'sc', 'sd', 'se', 'sg', 'si', 'sk',
    'sl', 'sm', 'sn', 'so', 'sq', 'sr', 'ss', 'st', 'su', 'sv', 'sw', 'ta',
    'te', 'tg', 'th', 'ti', 'tk', 'tl', 'tn', 'to', 'tr', 'ts', 'tt', 'tw',
    'ty', 'ug', 'uk', 'ur', 'uz', 've', 'vi', 'vo', 'wa', 'wo', 'xh', 'yi',
    'yo', 'za', 'zh', 'zu']


def fix_seed(seed=None):
    if seed is None:
        seed = time.time()

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return seed


def validate_or_assert(option_name, option_value, valid_options):
    assert option_value in valid_options, \
        f"{option_name!r} should be one of {valid_options!r}"


def get_meteor_jar(ver='1.5'):
    root = pathlib.Path(os.getenv('HOME')) / '.nmtpy' / 'meteor-data'
    jar = root / 'meteor-1.5.jar'
    assert jar.exists(), "METEOR not installed, please run 'nmtpy-install-extra'"
    return jar


def pbar(iterator, unit='it'):
    return tqdm(iterator, unit=unit, ncols=70, smoothing=0)


def load_pt_file(fname, device='cpu'):
    """Returns saved .(ck)pt file fields."""
    fname = str(pathlib.Path(fname).expanduser())
    data = torch.load(fname, map_location=device)
    if 'history' not in data:
        data['history'] = {}
    return data


def get_language(fname):
    """Heuristic to detect the language from filename components."""
    suffix = pathlib.Path(fname).suffix[1:]
    if suffix not in LANGUAGES:
        logger.info(f"Can not detect language from {fname}, fallback to 'en'")
        suffix = 'en'
    return suffix


def listify(l):
    """Encapsulate l with list[] if not."""
    return [l] if not isinstance(l, list) else l


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_local_args(d):
    return {k: v for k, v in d.items() if not k.startswith(('__', 'self'))}


def ensure_dirs(dirs):
    """Create a list of directories if not exists."""
    dirs = [pathlib.Path(d) for d in listify(dirs)]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def fopen(filename, key=None):
    """gzip,bzip2,xz,numpy aware file opening function."""
    assert '*' not in str(filename), "Glob patterns not supported in fopen()"

    filename = str(pathlib.Path(filename).expanduser())
    if filename.endswith('.gz'):
        return gzip.open(filename, 'rt')
    elif filename.endswith('.bz2'):
        return bz2.open(filename, 'rt')
    elif filename.endswith(('.xz', '.lzma')):
        return lzma.open(filename, 'rt')
    elif filename.endswith(('.npy', '.npz')):
        if filename.endswith('.npz'):
            assert key is not None, "No key= given for .npz file."
            return np.load(filename)[key]
        else:
            return np.load(filename)
    else:
        # Plain text
        return open(filename, 'r')


def readable_size(n):
    """Return a readable size string."""
    sizes = ['K', 'M', 'G']
    fmt = ''
    size = n
    for i, s in enumerate(sizes):
        nn = n / (1000 ** (i + 1))
        if nn >= 1:
            size = nn
            fmt = sizes[i]
        else:
            break
    return '%.2f%s' % (size, fmt)


def get_module_groups(layer_names):
    groups = set()
    for name in layer_names:
        if '.weight' in name:
            groups.add(name.split('.weight')[0])
        elif '.bias' in name:
            groups.add(name.split('.bias')[0])
    return sorted(list(groups))


def get_n_params(module):
    n_param_learnable = 0
    n_param_frozen = 0

    for param in module.parameters():
        if param.requires_grad:
            n_param_learnable += np.cumprod(param.data.size())[-1]
        else:
            n_param_frozen += np.cumprod(param.data.size())[-1]

    n_param_all = n_param_learnable + n_param_frozen
    return "# parameters: {} ({} learnable)".format(
        readable_size(n_param_all), readable_size(n_param_learnable))


def get_temp_file(delete=False, close=False):
    """Creates a temporary file under a folder."""
    root = pathlib.Path(os.environ.get('NMTPY_TMP', '/tmp'))
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    prefix = str(root / "nmtpytorch_{}".format(os.getpid()))
    t = tempfile.NamedTemporaryFile(
        mode='w', prefix=prefix, delete=delete)
    cleanup.register_tmp_file(t.name)
    if close:
        t.close()
    return t


def setup_experiment(opts, suffix=None, short=False, beat_platform=False):
    """Return a representative string for the experiment."""
    if not beat_platform:
        # subfolder is conf filename without .conf suffix
        opts.train['subfolder'] = pathlib.Path(opts.filename).stem

        # add suffix to subfolder name to keep experiment names shorter
        if suffix:
            opts.train['subfolder'] += "-{}".format(suffix)

        # Create folders
        folder = pathlib.Path(opts.train['save_path']) / opts.train['subfolder']
        folder.mkdir(parents=True, exist_ok=True)

    # Set random experiment ID
    run_id = time.strftime('%Y%m%d%H%m%S') + str(random.random())
    run_id = sha256(run_id.encode('ascii')).hexdigest()[:5]

    # Finalize
    model_type = opts.train['model_type'].lower()
    opts.train['exp_id'] = f'{model_type}-r{run_id}'
