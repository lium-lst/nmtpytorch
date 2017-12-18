# -*- coding: utf-8 -*-
import os
import bz2
import gzip
import lzma
import pathlib
import tempfile

import numpy as np
import torch
from tqdm import tqdm

from ..cleanup import cleanup

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


def pbar(iterator, unit='it'):
    return tqdm(iterator, unit=unit, ncols=70, smoothing=0)


def load_pt_file(fname):
    """Returns saved .(ck)pt file fields."""
    fname = str(pathlib.Path(fname).expanduser())
    data = torch.load(fname, map_location=lambda storage, loc: storage)
    if 'history' not in data:
        data['history'] = {}
    return data['model'], data['history'], data['opts']


def get_language(fname):
    suffix = pathlib.Path(fname).suffix[1:]
    assert suffix in LANGUAGES, \
        "Can not detect language from {}.".format(fname)
    return suffix


def listify(l):
    """Encapsulate l with list[] if not."""
    return [l] if not isinstance(l, list) else l


def flatten(l):
    return [item for sublist in l for item in sublist]


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


def force_symlink(origfile, linkname, relative=False):
    if relative:
        origfile = os.path.basename(origfile)
    try:
        os.symlink(origfile, linkname)
    except FileExistsError as e:
        os.unlink(linkname)
        os.symlink(origfile, linkname)


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
    return '%.1f%s' % (size, fmt)


def get_n_params(module):
    n_param_learnable = 0
    n_param_frozen = 0

    for param in module.parameters():
        if param.requires_grad:
            n_param_learnable += np.cumprod(param.data.size())[-1]
        else:
            n_param_frozen += np.cumprod(param.data.size())[-1]

    n_param_all = n_param_learnable + n_param_frozen
    return readable_size(n_param_all), readable_size(n_param_learnable)


def get_temp_file(suffix="", name=None, delete=False):
    """Creates a temporary file under /tmp."""
    if name:
        name = os.path.join("/tmp", name)
        t = open(name, "w")
        cleanup.register_tmp_file(name)
    else:
        _suffix = "_nmtpytorch_%d" % os.getpid()
        if suffix != "":
            _suffix += suffix
        t = tempfile.NamedTemporaryFile(mode='w', suffix=_suffix,
                                        delete=delete)
        cleanup.register_tmp_file(t.name)
    return t


def setup_experiment(opts, suffix=None):
    """Return a representative string for the experiment."""

    mopts = opts.model.copy()

    # Start with model name
    names = [opts.train['model_type'].lower()]

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

    if 'att_type' in mopts:
        names.append("att_%s" % mopts['att_type'])

    # Join so far
    name = '-'.join(names)

    # Append optimizer and learning rate
    name += '-%s_%.e' % (opts.train['optimizer'], opts.train['lr'])

    # Append batch size
    name += '-bs%d' % opts.train['batch_size']

    # Validation stuff (first: early-stop metric)
    name += '-%s' % opts.train['eval_metrics'].split(',')[0]

    if opts.train['eval_freq'] > 0:
        name += "-each%d" % opts.train['eval_freq']
    else:
        name += "-eachepoch"

    if opts.train['l2_reg'] > 0:
        name += "-l2_%.e" % opts.train['l2_reg']

    # Dropout parameter names can be different for each model
    dropouts = sorted([opt for opt in mopts if opt.startswith('dropout')])
    if len(dropouts) > 0:
        for dout in dropouts:
            _, layer = dout.split('_')
            if mopts[dout] > 0:
                name += "-do_%s_%.1f" % (layer, mopts[dout])

    # FIXME: We can't add everything to here so maybe we
    # need to let models to append their custom fields afterwards
    if mopts.get('tied_emb', False):
        name += "-%stied" % mopts['tied_emb']

    if mopts.get('simple_output', False):
        name += "-smpout"

    # Append seed
    name += "-s%d" % opts.train['seed']

    if suffix:
        name = "%s-%s" % (name, suffix)

    # Main folder is conf filename without .conf suffix i.e., nmt-en-de
    opts.train['subfolder'] = pathlib.Path(opts.filename).stem

    # Log file, runs start from 1, incremented if exists
    save_path = pathlib.Path(opts.train['save_path'])
    run_id = len(list((save_path / opts.train['subfolder']).glob(
        '%s.*.log' % name))) + 1

    # Save experiment ID
    opts.train['exp_id'] = '%s.%d' % (name, run_id)

    # Create folders
    (save_path / opts.train['subfolder']).mkdir(parents=True, exist_ok=True)
