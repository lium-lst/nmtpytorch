![nmtpytorch](docs/logo.png?raw=true "nmtpytorch")

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the PyTorch fork of [nmtpy](https://github.com/lium-lst/nmtpy),
a sequence-to-sequence framework which was originally a fork of
[dl4mt-tutorial](https://github.com/nyu-dl/dl4mt-tutorial).

The core parts of `nmtpytorch` depends on `numpy`, `torch` and `tqdm`.
For multimodal architectures, you also need to install `torchvision` which
is used to integrate pre-trained CNN models.

`nmtpytorch` is developed and tested on Python 3.6 and will not support
Python 2.x whatsoever.

## Installation

We ship [subword-nmt](https://github.com/rsennrich/subword-nmt) and METEOR
paraphrase files as submodules in order to track their updates when
necessary. Besides these METEOR v1.5 JAR, `multi-bleu.perl` and COCO
evaluation tools `cocoeval` are directly included within the source tree.

Run the following command to recursively fetch the repository
including those submodules:

```
git clone --recursive https://github.com/lium-lst/nmtpytorch.git
```

Install using `develop` mode:

```
python setup.py develop
```

## Usage Example

A [sample NMT configuration](examples/multi30k-en-de-bpe10k.conf) for
English-to-German Multi30k is provided which covers nearly all of the `[train]`
and `[model]` specific options to `NMT`.

After creating a configuration file for your own dataset that suits your need,
you can run the following command to start training:

```
nmtpy train -C <config file>
```

It is possible to override any configuration option through the command-line:

```
nmtpy train -C <config file> train.<opt>:<val> model.<opt>:<val> ...
```

## Release Notes

### v1.2 (20/02/2018)

 - You can now use `$HOME` and `$USER` in your configuration files.
 - Fixed an overflow error that would cause NMT with more than 255 tokens to fail.
 - METEOR worker process is now correctly killed after validations.
 - Many runs of an experiment are now suffixed with a unique random string instead of incremental integers to avoid race conditions in cluster setups.
 - Replaced `utils.nn.get_network_topology()` with a new `Topology` [class](nmtpytorch/utils/topology.py) that will parse the `direction` string of the model in a more smart way.
 - If `CUDA_VISIBLE_DEVICES` is set, the `GPUManager` will always honor it.
 - Dropped creation of temporary/advisory lock files under `/tmp` for GPU reservation.
 - Time measurements during training are now structered into batch overhead, training and evaluation timings.
 - **Datasets**
   - Added `TextDataset` for standalone text file reading.
   - Added `OneHotDataset`, a variant of `TextDataset` where the sequences are not prefixed/suffixed with `<bos>` and `<eos>` respectively.
   - Added experimental `MultiParallelDataset` that merges an arbitrary number of parallel datasets together.
 - **nmtpy translate**
   - `.nodbl` and `.nounk` suffixes are now added to output files for `--avoid-double` and `--avoid-unk` arguments respectively.
   - A model-agnostic enough `beam_search()` is now separated out into its own file `nmtpytorch/search.py`.
   - `max_len` default is increased to 200.

### v1.1 (25/01/2018)

 - New experimental `Multi30kDataset` and `ImageFolderDataset` classes
 - `torchvision` dependency added for CNN support
 - `nmtpy-coco-metrics` now computes one METEOR without `norm=True`
 - Mainloop mechanism is completely refactored with **backward-incompatible**
   configuration option changes for `[train]` section:
    - `patience_delta` option is removed
    - Added `eval_batch_size` to define batch size for GPU beam-search during training
    - `eval_freq` default is now `3000` which means per `3000` minibatches
    - `eval_metrics` now defaults to `loss`. As before, you can provide a list
      of metrics like `bleu,meteor,loss` to compute all of them and early-stop
      based on the first
    - Added `eval_zero (default: False)` which tells to evaluate the model
      once on dev set right before the training starts. Useful for sanity
      checking if you fine-tune a model initialized with pre-trained weights
    - Removed `save_best_n`: we no longer save the best `N` models on dev set
      w.r.t. early-stopping metric
    - Added `save_best_metrics (default: True)` which will save best models
      on dev set w.r.t each metric provided in `eval_metrics`. This kind of
      remedies the removal of `save_best_n`
    - `checkpoint_freq` now to defaults to `5000` which means per `5000`
      minibatches.
    - Added `n_checkpoints (default: 5)` to define the number of last
      checkpoints that will be kept if `checkpoint_freq > 0` i.e. checkpointing enabled
  - Added `ExtendedInterpolation` support to configuration files:
    - You can now define intermediate variables in `.conf` files to avoid
      typing same paths again and again. A variable can be referenced
      from within its **section** using `tensorboard_dir: ${save_path}/tb` notation
      Cross-section references are also possible: `${data:root}` will be replaced
      by the value of the `root` variable defined in the `[data]` section.
  - Added `-p/--pretrained` to `nmtpy train` to initialize the weights of
    the model using another checkpoint `.ckpt`.
  - Improved input/output handling for `nmtpy translate`:
    - `-s` accepts a comma-separated test sets **defined** in the configuration
      file of the experiment to translate them at once. Example: `-s val,newstest2016,newstest2017`
    - The mutually exclusive counterpart of `-s` is `-S` which receives a
      single input file of source sentences.
    - For both cases, an output prefix **should now be** provided with `-o`.
      In the case of multiple test sets, the output prefix will be appended
      the name of the test set and the beam size. If you just provide a single file with `-S`
      the final output name will only reflect the beam size information.
 - Two new arguments for `nmtpy-build-vocab`:
    - `-f`: Stores frequency counts as well inside the final `json` vocabulary
    - `-x`: Does not add special markers `<eos>,<bos>,<unk>,<pad>` into the vocabulary

#### Layers/Architectures

 - Added `Fusion()` layer to `concat,sum,mul` an arbitrary number of inputs
 - Added *experimental* `ImageEncoder()` layer to seamlessly plug a VGG or ResNet
   CNN using `torchvision` pretrained models
 - `Attention` layer arguments improved. You can now select the bottleneck
   dimensionality for MLP attention with `att_bottleneck`. The `dot`
   attention is **still not tested** and probably broken.

New layers/architectures:

 - Added **AttentiveMNMT** which implements modality-specific multimodal attention
   from the paper [Multimodal Attention for Neural Machine Translation](https://arxiv.org/abs/1609.03976)
 - Added **ShowAttendAndTell** [model](http://www.jmlr.org/proceedings/papers/v37/xuc15.pdf)

Changes in **NMT**:

  - `dec_init` defaults to `mean_ctx`, i.e. the decoder will be initialized
    with the mean context computed from the source encoder
  - `enc_lnorm` which was just a placeholder is now removed since we do not
    provided layer-normalization for now
  - Beam Search is completely moved to GPU

### Initial Release v1.0 (18/12/2017)

The initial release aims to be (as much as) feature compatible with respect
to the latest `nmtpy` with some important changes as well.

#### New TensorBoard Support

If you would like to monitor training progress, you may want to install
[tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch). Note that
you will also need to install the actual TensorBoard server which is shipped
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

![tensorboard](docs/tensorboard.png?raw=true "tensorboard")


#### A Single Command-Line Interface

Instead of shipping several tools for training, rescoring, translating, etc.
here we provide a single command-line interface `nmtpy` which implements
three subcommands `train`, `translate` and `resume`.

**`nmtpy train`**

```
usage: nmtpy train [-h] -C CONFIG [-s SUFFIX] [overrides [overrides ...]]

positional arguments:
  overrides             (section).key:value overrides for config

optional arguments:
  -h, --help            show this help message and exit
  -C CONFIG, --config CONFIG
                        Experiment configuration file
  -s SUFFIX, --suffix SUFFIX
                        Optional experiment suffix.

```

**`nmtpy translate`**
```
usage: nmtpy translate [-h] [-n] -s SPLITS [-b BATCH_SIZE] [-k BEAM_SIZE]
                       [-m MAX_LEN] [-p] [-u] [-d DEVICE] [-e]
                       models [models ...]

positional arguments:
  models                Saved model/checkpoint file(s)

optional arguments:
  -h, --help            show this help message and exit
  -n, --disable-filters
                        Disable text filters given in config.
  -s SPLITS, --splits SPLITS
                        Comma separated splits to translate
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size for beam-search
  -k BEAM_SIZE, --beam-size BEAM_SIZE
                        Beam size for beam-search
  -m MAX_LEN, --max-len MAX_LEN
                        Maximum sequence length
  -p, --avoid-double    Suppress previous token probs
  -u, --avoid-unk       Suppress <unk> generation
  -d DEVICE, --device DEVICE
                        Select GPU device(s)
  -e, --ensemble        Enable ensembling for multiple models.
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

We recommend you to take a look at the provided sample
[configuration](examples/multi30k-en-de-bpe10k.conf) to have an idea about the file format.

#### Training a Model

We still provide a single, model-agnostic `mainloop` that handles everything
necessary to train, validate and early-stop a model.

#### Defining a Model

You just need to create a new file under `nmtpytorch/models` and define a
`class` by deriving it from `nn.Module`. The name of this new `class` will be the
`model_type` that needs to be written inside your configuration file. The next
steps are to:

 - Parse model options passed from the configuration file in `__init__()`
 - Define layers inside `setup()`: Each `nn.Module` object should be assigned
   as an attribute of the model (i.e. `self.encoder = ...`) in order for
   PyTorch to work correctly.
 - Create and store relevant dataset objects in `load_data()`
 - Define `compute_loss()` which takes a data iterator and
   computes the loss over it. This method is used for dev set perplexities.
 - Set `aux_loss` attribute for an additional loss term.
 - Define `forward()` which takes a dictionary with keys as data sources and
   returns the batch training loss. This is the method called from the `mainloop`
   during training.

Feel free to copy the methods from `NMT` if you do not need to modify
some of them.

#### Provided Models

Currently we only provide a **Conditional GRU NMT** [implementation](nmtpytorch/models/nmt.py)
with Bahdanau-style attention in decoder.

**NOTE**: We recommend limiting the number of tokens in the target vocabulary
by defining `max_trg_len` in the `[model]` section of your configuration file
to avoid GPU out of memory errors for very large vocabularies. This is caused
by the fact that the gradient computation for a batch with very long sequences
occupies a large amount of memory unless the loss layer is implemented differently.
