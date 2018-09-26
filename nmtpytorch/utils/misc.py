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
    return data['model'], data['history'], data['opts']


def get_language(fname):
    suffix = pathlib.Path(fname).suffix[1:]
    if suffix not in LANGUAGES:
        logger.info("Can not detect language from {}.".format(fname))
        return None
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


def get_temp_file(delete=False):
    """Creates a temporary file under a folder."""
    root = pathlib.Path(os.environ.get('NMTPY_TMP', '/tmp'))
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    prefix = str(root / "nmtpytorch_{}".format(os.getpid()))
    t = tempfile.NamedTemporaryFile(
        mode='w', prefix=prefix, delete=delete)
    cleanup.register_tmp_file(t.name)
    return t


def setup_experiment(opts, suffix=None, short=False):
    """Return a representative string for the experiment."""

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

    names = []

    mopts = opts.model.copy()

    # Start with general ones
    if 'enc_type' in mopts:
        names.append("enc%d%s%d" % (mopts.get('n_encoders', 1),
                                    mopts['enc_type'].upper(),
                                    mopts.pop('enc_dim')))
    if 'dec_type' in mopts:
        names.append("dec%d%s%d" % (mopts.get('n_decoders', 1),
                                    mopts['dec_type'].upper(),
                                    mopts.pop('dec_dim')))
    for k in sorted(mopts):
        if k.endswith("_dim"):
            names.append('%s%d' % (k.split('_')[0], mopts[k]))

    # Append optimizer and learning rate
    names.append('%s_%.e' % (opts.train['optimizer'], opts.train['lr']))

    if opts.train['l2_reg'] > 0:
        names.append('l2_%.e' % opts.train['l2_reg'])

    # Dropout parameter names can be different for each model
    dropouts = sorted([opt for opt in mopts if opt.startswith('dropout')])
    for dout in dropouts:
        if mopts[dout] > 0:
            parts = dout.split('_')
            if len(parts) == 2:
                names.append("do_%s_%.1f" % (parts[1], mopts[dout]))
            else:
                names.append("do%.1f" % mopts[dout])

    # If short names requested, we stop here
    if short:
        name = "-".join(names)
        opts.train['exp_id'] = '%s-r%s' % (name, run_id)
        return

    # Continue with other stuff
    if 'att_type' in mopts:
        names.append('att_{}'.format(mopts['att_type']))

    if 'fusion_type' in mopts:
        names.append('ctx_{}'.format(mopts['fusion_type']))

    # Append batch size
    names.append('bs{}'.format(opts.train['batch_size']))

    # Validation stuff (first: early-stop metric)
    names.append(opts.train['eval_metrics'].split(',')[0])

    if opts.train['eval_freq'] > 0:
        names.append("each{}".format(opts.train['eval_freq']))
    else:
        names.append("eachepoch")

    # FIXME: We can't add everything to here so maybe we
    # need to let models to append their custom fields afterwards
    if mopts.get('tied_emb', False):
        names.append("{}tied".format(mopts['tied_emb']))

    if mopts.get('dec_init', False):
        names.append("di_{}".format(mopts['dec_init'].replace('_', '')))

    # Finalize
    opts.train['exp_id'] = '{}-{}-r{}'.format(
        opts.train['model_type'].lower(), "-".join(names), run_id)
