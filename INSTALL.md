# Installation Guide

As of 1.4.0, `nmtpytorch` supports only GPU and strictly requires the
following packages:

  - Python 3.6
  - Torch 0.3.1
  - A working Java Runtime Environment (i.e. `java` in `$PATH`) for METEOR.


We directly carry `multi-bleu.perl` and COCO evaluation tools `cocoeval` within the source tree.

**NOTE**: In order to get the necessary METEOR files, after installing `nmtpytorch`, you
need to run `nmtpy-install-extra` once so that they are installed under
`${HOME}/.nmtpy`.

## Anaconda Python (Recommended)

Replace below `<VER>` with 8, 9.0 or 9.1 depending on your nVidia driver version:

```
$ conda update --all
$ git clone --recursive https://github.com/lium-lst/nmtpytorch.git
$ conda env create -f nmtpytorch/environment-cuda<VER>.yml
```

This should create a `conda` environment called `nmtpy` with all of the
dependencies installed inside.

## Virtualenv

```
$ python3.6 -m venv nmtpyvenv
$ cd nmtpyvenv
$ source bin/activate
(nmtpyvenv) $ git clone --recursive https://github.com/lium-lst/nmtpytorch.git src/
(nmtpyvenv) $ pip install -e src/
```

NOTE: This will bring CUDA8 version of `torch` since it's the default package in `PyPI`.
If you want CUDA9 or CUDA9.1 instead, install `torch` first inside the virtual environment
and then call `pip install -e src/`.

## Modifying the Code Tree

Both methods install `nmtpytorch` in the so-called **editable** mode. This
means that the Python interpreter will see the changes that you'll made inside the cloned folder.
This has the advantage of avoiding reinstallations once you modify some model,
add new features or a new model.
