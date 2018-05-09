# Installation Guide

As of 1.4.0, `nmtpytorch` supports only GPU and strictly requires the
following packages:

  - Python 3.6
  - Torch 0.3.1
  - A working Java Runtime Environment (i.e. `java` in `$PATH`) for METEOR.


We directly carry METEOR v1.5, `multi-bleu.perl` and COCO
evaluation tools `cocoeval` within the source tree. Moreover, we ship
[subword-nmt](https://github.com/rsennrich/subword-nmt) and
METEOR paraphrase files as GIT submodules in order to track their updates when
necessary. Since paraphrase files are large, cloning the repository may take
some time.

## Anaconda Python (Recommended)

```
$ conda update --all
$ git clone --recursive https://github.com/lium-lst/nmtpytorch.git
$ conda env create -f nmtpytorch/environment.yml
```

This should create a `conda` environment called `nmtpy` with all of the
dependencies installed inside.

NOTE: The `environment.yml` file specifically installs `torch 0.3.1` with CUDA9.
If you want to use CUDA8, you can modify the file to install instead the
CUDA8 `.whl`.

## Virtualenv

```
$ python3.6 -m venv nmtpyvenv
$ cd nmtpyvenv
$ source bin/activate
(nmtpyvenv) $ git clone --recursive https://github.com/lium-lst/nmtpytorch.git src/
(nmtpyvenv) $ pip install -e src/
```

NOTE: This in contrary to `conda` installation, will bring CUDA8 version of
`torch` since it's the default package in `PyPI`. I couldn't the correct way
of writing a dependency of CUDA9 version of `torch` inside `setup.py`. So feel
free the remove/uninstall/install `torch 0.3.1` with the CUDA version that suits you.

## Modifying the source codes

Both methods install `nmtpytorch` in the so-called **editable** mode. This
means that the Python interpreter will see the changes that you'll made inside the cloned folder.
This has the advantage of avoiding reinstallations once you modify some model,
add new features or a new model.
