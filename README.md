![nmtpytorch](https://github.com/lium-lst/nmtpytorch/blob/master/docs/logo.png?raw=true "nmtpytorch")

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

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

`nmtpytorch` currently requires `python>=3.6`, `torch==0.3.1` and a GPU to work.
We are not planning to support Python 2.x but it will be updated to work with the
newer versions of `torch` and also on CPU.

### pip

You can install `nmtpytorch` from `PyPI` using `pip` (or `pip3` depending on your
operating system and environment):

```
$ pip install nmtpytorch
```

This will automatically fetch and install the dependencies as well. For the `torch`
dependency it will specifically install the `torch 0.3.1` package from `PyPI` that
ships `CUDA 8.0` within. If instead you want to use a newer version of `CUDA`,
you can uninstall the `torch` package manually afterwards and install another `0.3.1`
package from [here](https://pytorch.org/get-started/previous-versions/).

### conda

We provide an `environment.yml` file in the repository that you can use to create
a ready-to-use anaconda environment for `nmtpytorch`:

```
$ conda update --all
$ git clone https://github.com/lium-lst/nmtpytorch.git
$ conda env create -f nmtpytorch/environment.yml
```

Unlike the `pip` method, this environment explicitly installs the `CUDA 9.0`
version of `torch 0.3.1` and enables editable mode similar to the development
mode explained below.

### Development Mode

For continuous development and testing, it is sufficient to run `python setup.py develop`
in the root folder of your GIT checkout. From now on, all modifications to the source
tree are directly taken into account without requiring reinstallation.

### METEOR Installation

After the above installation steps, you finally need to run `nmtpy-install-extra`
in order to fetch and store METEOR related files in your `${HOME}/.nmtpy` folder.
This step is only required once.

## Documentation

We currently only provide some preliminary documentation in our [wiki](https://github.com/lium-lst/nmtpytorch/wiki).

## Release Notes

See [NEWS.md](NEWS.md).

## Usage Example

After creating a configuration file for your own dataset that suits your need,
you can run the following command to start training:

```
nmtpy train -C <config file>
```

It is possible to override any configuration option through the command-line:

```
nmtpy train -C <config file> train.<opt>:<val> model.<opt>:<val> ...
```

## TensorBoard Support

You will need to install the actual TensorBoard server which is shipped
within Tensorflow in order to launch the visualization server.

Once the dependencies are installed, you need to define a log directory for
TensorBoard in the `configuration` file of your experiment to enable
TensorBoard logging. The logging frequency is the same as terminal logging
frequency, defined by `train.disp_freq` option (default: 30 batches).

```
[train]
..
tensorboard_dir: ~/tb_dir
```

![tensorboard](https://github.com/lium-lst/nmtpytorch/blob/master/docs/tensorboard.png?raw=true "tensorboard")


## A Single Command-Line Interface

Instead of shipping several tools for training, rescoring, translating, etc.
here we provide a single command-line interface `nmtpy` which implements
three subcommands `train`, `translate` and `test`.

**`nmtpy train`**

```
usage: nmtpy train [-h] -C CONFIG [-s SUFFIX] [-S] [overrides [overrides ...]]

positional arguments:
  overrides             (section).key:value overrides for config

optional arguments:
  -h, --help            show this help message and exit
  -C CONFIG, --config CONFIG
                        Experiment configuration file
  -s SUFFIX, --suffix SUFFIX
                        Optional experiment suffix.
  -S, --short           Use short experiment id in filenames.
```

**`nmtpy translate`**
```
usage: nmtpy translate [-h] [-n] [-b BATCH_SIZE] [-k BEAM_SIZE] [-m MAX_LEN]
                       [-a LP_ALPHA] [-d DEVICE_ID] (-s SPLITS | -S SOURCE) -o
                       OUTPUT
                       models [models ...]

positional arguments:
  models                Saved model/checkpoint file(s)

optional arguments:
  -h, --help            show this help message and exit
  -n, --disable-filters
                        Disable eval_filters given in config
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size for beam-search
  -k BEAM_SIZE, --beam-size BEAM_SIZE
                        Beam size for beam-search
  -m MAX_LEN, --max-len MAX_LEN
                        Maximum seq. limit (Default: 200)
  -a LP_ALPHA, --lp-alpha LP_ALPHA
                        Apply length-penalty (Default: 0.)
  -d DEVICE_ID, --device-id DEVICE_ID
                        Select GPU device(s)
  -s SPLITS, --splits SPLITS
                        Comma separated splits from config file
  -S SOURCE, --source SOURCE
                        Comma-separated key:value pairs to provide new inputs.
  -o OUTPUT, --output OUTPUT
                        Output filename prefix
```

#### Experiment Configuration

The INI-style experiment configuration file format is slightly updated to
allow for future multi-task, multi-lingual setups in terms of data description.

Model-agnostic options are defined in `[train]` section while the options
that will be consumed by the model itself are defined in `[model]`.

An arbitrary number of parallel corpora with multiple languages can be defined
in `[data]` section. Note that you **need** to define at least
`train_set` and `val_set` datasets in this section for the training and
early-stopping to work correctly.
