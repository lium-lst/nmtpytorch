![nmtpytorch](https://github.com/lium-lst/nmtpytorch/blob/master/docs/logo.png?raw=true "nmtpytorch")

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

Click [here](https://join.slack.com/t/nmtpy/shared_invite/enQtNDQ5NTY2ODU1MzI5LTBkMzZjNjYyZDZiYjk0NjNmYzRmMzY2MWVjZWYyMDRhNTFiOTljY2UyYWVjODUyZWQ1YWExMWEzZmU2MmFkZmY) to join to our Slack channel for questions.

`nmtpytorch` allows training of various end-to-end neural architectures including
but not limited to neural machine translation, image captioning and automatic
speech recognition systems. The initial codebase was in `Theano` and was
inspired from the famous [dl4mt-tutorial](https://github.com/nyu-dl/dl4mt-tutorial)
codebase.

`nmtpytorch` is mainly developed by the **Language and Speech Team** of **Le Mans University** but
receives valuable contributions from the [Grounded Sequence-to-sequence Transduction Team](https://github.com/srvk/jsalt-2018-grounded-s2s)
of *Frederick Jelinek Memorial Summer Workshop 2018*:

Loic Barrault, Ozan Caglayan, Amanda Duarte, Desmond Elliott, Spandana Gella, Nils Holzenberger,
Chirag Lala, Jasmine (Sun Jae) Lee, Jindřich Libovický, Pranava Madhyastha,
Florian Metze, Karl Mulligan, Alissa Ostapenko, Shruti Palaskar, Ramon Sanabria, Lucia Specia and Josiah Wang.

If you use **nmtpytorch**, you may want to cite the following [paper](https://ufal.mff.cuni.cz/pbml/109/art-caglayan-et-al.pdf):
```
@article{nmtpy2017,
  author    = {Ozan Caglayan and
               Mercedes Garc\'{i}a-Mart\'{i}nez and
               Adrien Bardet and
               Walid Aransa and
               Fethi Bougares and
               Lo\"{i}c Barrault},
  title     = {NMTPY: A Flexible Toolkit for Advanced Neural Machine Translation Systems},
  journal   = {Prague Bull. Math. Linguistics},
  volume    = {109},
  pages     = {15--28},
  year      = {2017},
  url       = {https://ufal.mff.cuni.cz/pbml/109/art-caglayan-et-al.pdf},
  doi       = {10.1515/pralin-2017-0035},
  timestamp = {Tue, 12 Sep 2017 10:01:08 +0100}
}
```

## Installation

`nmtpytorch` currently requires `python>=3.6` and `torch>=0.4.1`.
We are not planning to support Python 2.x.

**IMPORTANT:** After installing `nmtpytorch`, you **need** to run `nmtpy-install-extra`
to download METEOR related files into your `${HOME}/.nmtpy` folder.
This step is only required once.

### pip

You can install `nmtpytorch` from `PyPI` using `pip` (or `pip3` depending on your
operating system and environment):

```
$ pip install nmtpytorch
```

This will automatically fetch and install the dependencies as well. For the `torch`
dependency it will specifically install the `torch 1.0.0` package from `PyPI` that
ships `CUDA 9.0` within.

### conda

We provide an `environment.yml` file in the repository that you can use to create
a ready-to-use anaconda environment for `nmtpytorch`:

```
$ conda update --all
$ git clone https://github.com/lium-lst/nmtpytorch.git
$ conda env create -f nmtpytorch/environment.yml
```

### Development Mode

For continuous development and testing, it is sufficient to run `python setup.py develop`
in the root folder of your GIT checkout. From now on, all modifications to the source
tree are directly taken into account without requiring reinstallation.

## Documentation

We currently only provide some preliminary documentation in our [wiki](https://github.com/lium-lst/nmtpytorch/wiki).

## Release Notes

See [NEWS.md](NEWS.md).
