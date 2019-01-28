Automatic Speech Recognition (ASR)
--

Two example configuration files for character level and subword level
ASR systems.

First, we will briefly specify how to organize the input and output
features and labels to train an ASR system.

## Preparing Kaldi features

Right now nmtpytorch only supports Kaldi feature files namely `.ark` and `.scp`
files along with a special folder structure. Let's assume that all speech related
files are under `/path/to/data/speech`. The data should be further organized
as follows:

  - A subfolder for each dataset split, i.e. train, dev, test and other sets.
    - An `.scp` file which should be called `feats_local.scp`
    - A `segments.len` file listing the number of frames per utterance on each line.
  - The `.ark` files referred to by the `feats_local.scp` should be uncompressed
    using `copy-feats` utility from Kaldi.

We will further detail how to ge
