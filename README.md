![nmtpytorch](https://github.com/lium-lst/nmtpytorch/blob/master/docs/logo.png?raw=true "nmtpytorch")

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the PyTorch fork of [nmtpy](https://github.com/lium-lst/nmtpy),
a sequence-to-sequence framework which was originally a fork of
[dl4mt-tutorial](https://github.com/nyu-dl/dl4mt-tutorial).

`nmtpytorch` is developed and tested on Python 3.6 and will not support
Python 2.x whatsoever.

## Citation

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
