## Release Notes

### v1.3.2 (02/05/2018)

  - Updates to `ShowAttendAndTell` model.

### v1.3.1 (01/05/2018)

  - Removed old `Multi30kDataset`.
  - Sort batches by source sequence length instead of target.
  - Fix `ShowAttendAndTell` model. It should now work.

### v1.3 (30/04/2018)

 - Added `Multi30kRawDataset` for training end-to-end systems from raw images as input.
 - Added `NumpyDataset` to read `.npy/.npz` tensor files as input features.
 - You can now pass `-S` to `nmtpy train` to produce shorter experiment files with not all the hyperparameters in file name.
 - New post-processing filter option `de-spm` for Google SentencePiece (SPM) processed files.
 - `sacrebleu` is now a dependency as it is now accepted as an early-stopping metric.
 It only makes sense to use it with SPM processed files since they are detokenized
 once post-processed.
 - Added `sklearn` as a dependency for some metrics.
 - Added `momentum` and `nesterov` parameters to `[train]` section for SGD.
 - `ImageEncoder` layer is improved in many ways. Please see the code for further details.
 - Added unmerged upstream [PR](https://github.com/pytorch/pytorch/pull/5297/files) for `ModuleDict()` support.
 - `METEOR` will now fallback to English if language can not be detected from file suffixes.
 - `-f` now produces a separate numpy file for token frequencies when building vocabulary files with `nmtpy-build-vocab`.
 - Added new command `nmtpy test` for non beam-search inference modes.
 - Removed `nmtpy resume` command and added `pretrained_file` option for `[train]` to initialize model weights from a checkpoint.
 - Added `freeze_layers` option for `[train]` to give comma-separated list of layer name prefixes to freeze.
 - Improved seeding: seed is now printed in order to reproduce the results.
 - Added IPython notebook for attention visualization.
 - **Layers**
   - New shallow `SimpleGRUDecoder` layer.
   - `TextEncoder`: Ability to set `maxnorm` and `gradscale` of embeddings and work with or without sorted-length batches.
   - `ConditionalDecoder`: Make it work with GRU/LSTM, allow setting `maxnorm/gradscale` for embeddings.
   - `ConditionalMMDecoder`: Same as above.
 - **nmtpy translate**
   - `--avoid-double` and `--avoid-unk` removed for now.
   - Added Google's length penalty normalization switch `--lp-alpha`.
   - Added ensembling which is enabled automatically if you give more than 1 model checkpoints.
 - New machine learning metric wrappers in `utils/ml_metrics.py`:
   - Label-ranking average precision `lrap`
   - Coverage error
   - Mean reciprocal rank

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
