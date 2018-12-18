## Release Notes

This release supports Pytorch >= 0.4.1 including the recent 1.0 release. The relevant
`setup.py` and `environment.yml` files will default to `1.0.0` installation.

### v4.0.0 (18/12/2018)
 - **Critical**: `NumpyDataset` now returns tensors of shape `HxW, N, C` for 3D/4D convolutional features, `1, N, C` for 2D feature files. Models should be adjusted to adapt to this new shaping.
 - An `order_file` per split (`ord: path/to/txt file with integer per line`) can be given from the configurations to change the feature order of numpy tensors to flexibly revert, shuffle, tile, etc. them.
 - Better dimension checking to ensure that everything is OK.
 - Added `LabelDataset` for single label input/outputs with associated `Vocabulary` for integer mapping.
 - Added `handle_oom=(True|False)` argument for `[train]` section to recover from **GPU out-of-memory (OOM)** errors during training. This is disabled by default, you need to enable it from the experiment configuration file. Note that it is still possible to get an OOM during validation perplexity computation. If you hit that, reduce the `eval_batch_size` parameter.
 - Added `de-hyphen` post-processing filter to stitch back the aggressive hyphen splitting of Moses during early-stopping evaluations.
 - Added optional projection layer and layer normalization to `TextEncoder`.
 - Added `enc_lnorm, sched_sampling` options to `NMT` to enable layer normalization for encoder and use **scheduled sampling** at a given probability.
 - `ConditionalDecoder` can now be initialized with max-pooled encoder states or the last state as well.
 - You can now experiment with different decoders for `NMT` by changing the `dec_variant` option.
 - Collect all attention weights in `self.history` dictionary of the decoders.
 - Added **n-best** output to `nmtpy translate` with the argument `-N`.
 - Changed the way `-S` works for `nmtpy translate`. Now you need to give the split name with `-s` all the time but `-S` is used to override the input data sources defined for that split in the configuration file.
 - Removed decoder-initialized multimodal NMT `MNMTDecInit`. Same functionality exists within the `NMT` model by using the model option `dec_init=feats`.
 - **New model MultimodalNMT:** that supports encoder initialization, decoder initialization, both, concatenation of embeddings with visual features, prepending and appending. This model covers almost all the models from [LIUM-CVC's WMT17 multimodal systems](https://arxiv.org/abs/1707.04481) except the multiplicative interaction variants such as `trgmul`.
 - **New model MultimodalASR:** encoder-decoder initialized ASR model. See the [paper](https://arxiv.org/abs/1811.03865)
 - **New Model AttentiveCaptioning:** Similar but not an exact reproduction of show-attend-and-tell, it uses feature files instead of raw images.
 - **New model AttentiveMNMTFeaturesFA:** [LIUM-CVC's WMT18 multimodal system](https://arxiv.org/abs/1809.00151) i.e. filtered attention
 - **New (experimental) model NLI:** A simple LSTM-based NLI baseline for [SNLI](https://nlp.stanford.edu/projects/snli/) dataset:
    - `direction` should be defined as `direction: pre:Text, hyp:Text -> lb:Label`
    - `pre, hyp` and `lb` keys point to plain text files with one sentence per line. A vocabulary should be constructed even for the labels to fit the nmtpy architecture.
    - `acc` should be added to `eval_metrics` to compute accuracy.

### v3.0.0 (05/10/2018)
Major release that brings support for **Pytorch 0.4** and drops support for **0.3**.

Training and testing on **CPUs** are now supported thanks to easier device
semantics of Pytorch 0.4: just give `-d cpu` to `nmtpy` to switch to CPU mode.
NOTE: Training on CPUs is only logical for debugging, otherwise it's very slow.
  - NOTE: `device_id` is no longer a configuration option. It should be removed
  from your old configurations.
  - Multi-GPU is not supported. Always restrict to single GPU using
    `CUDA_VISIBLE_DEVICES` environment variable.

You can now override the config options used to train a model during
inference: Example: `nmtpy translate (...) -x model.att_temp:0.9`

`nmtpy train` now detects invalid/old `[train]` options and refuses to
train the model.

**New sampler:** `ApproximateBucketBatchSampler`
Similar to the default `BucketBatchSampler` but more efficient for sparsely
distributed sequence-lengths as in speech recognition. It bins similar-length
items to buckets. It no longer guarantees that the batches are completely
made of same-length sequences so **care has to be taken in the encoders**
to support packing/padding/masking. `TextEncoder` already does this automatically
while speech encoder `BiLSTMp` does not care.

**EXPERIMENTAL**: You can decode an ASR system using the approximate sampler
although the model does not take care of the padded positions (a warning
will be printed at each batch).
The loss is 0.2% WER for a specific dataset that we tried. So although the computations
in the encoder becomes noisy and not totally correct, the model can handle
this noise quite robustly:

`$ nmtpy translate -s val -o hyp -x model.sampler_type:approximate best_asr.ckpt`

This type of batching cuts ASR decoding time almost by a factor of 2-3.

#### Other changes
  - Vocabularies generated by `nmtpy-build-vocab` now contains frequency
    information as well. The code is backward-compatible with old vocab files.
  - `Batch` objects should now be explicitly moved to the allocated device
    using `.device()` method. See `mainloop.py` and `test_performance()` from
    the `NMT` model.
  - Training no longer shows the cached GPU allocation from `nvidia-smi` output
    as it was in the end a hacky thing to call `nvidia-smi` periodically. We
    plan to use `torch.cuda.*` to get an estimate on memory consumption.
  - NOTE: Multi-process data loading is temporarily disabled as it was
    crashing from time to time so `num_workers > 0` does not have an effect
    in this release.
  - `Attention` is separated into `DotAttention` and `MLPAttention` and a
    convenience function `get_attention()` is provided to select between them
    during model construction.
  - `get_activation_fn()` should be used to select between non-linearities
    dynamically instead of doing `getattr(nn.functional, activ)`. The latter
    will not work for `tanh` and `sigmoid` in the next Pytorch releases.
  - Simplification: `ASR` model is now derived from `NMT`.


### v2.0.0 (26/09/2018)
  - Ability to install through `pip`.
  - Advanced layers are now organized into subfolders.
  - New basic layers: Convolution over sequence, MaxMargin.
  - New attention layers: Co-attention, multi-head attention, hierarchical attention.
  - New encoders: Arbitrary sequence-of-vectors encoder, BiLSTMp speech feature encoder.
  - New decoders: Multi-source decoder, switching decoder, vector decoder.
  - New datasets: Kaldi dataset (.ark/.scp reader), Shelve dataset, Numpy sequence dataset.
  - Added learning rate annealing: See `lr_decay*` options in `config.py`.
  - Removed subword-nmt and METEOR files from repository. We now depend on
    the PIP package for subword-nmt. For METEOR, `nmtpy-install-extra` should
    be launched after installation.
  - More multi-task and multi-input/output `translate` and `training` regimes.
  - New early-stopping metrics: Character and word error rate (cer,wer) and ROUGE (rouge).
  - Curriculum learning option for the `BucketBatchSampler`, i.e. length-ordered batches.
  - New models:
     - ASR: Listen-attend-and-spell like automatic speech recognition
     - Multitask*: Experimental multi-tasking & scheduling between many inputs/outputs.

### v1.4.0 (09/05/2018)
  - Add `environment.yml` for easy installation using `conda`. You can now
  create a ready-to-use `conda` environment by just calling `conda env create -f environment.yml`.
  - Make `NumpyDataset` memory efficient by keeping `float16` arrays as they are
  until batch creation time.
  - Rename `Multi30kRawDataset` to `Multi30kDataset` which now supports both
  raw image files and pre-extracted visual features file stored as `.npy`.
  - Add CNN feature extraction script under `scripts/`.
  - Add doubly stochastic attention to `ShowAttendAndTell` and multimodal NMT.
  - New model `MNMTDecinit` to initialize decoder with auxiliary features.
  - New model `AMNMTFeatures` which is the attentive MMT but with features file
  instead of end-to-end feature extraction which was memory hungry.

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
