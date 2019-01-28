Automatic Speech Recognition (ASR)
--

Two example configuration files for character level and subword level
ASR systems.

First, we will briefly specify how to organize the input and output
features and labels to train an ASR system.

## Preparing Kaldi features

Right now nmtpytorch only supports Kaldi feature files namely `.ark` and `.scp`
files along with a special folder structure. Let's assume that all speech related
files are under `~/data/swbd`:
  - Each train/test set split should have a corresponding subfolder with the following files:
    - feats.scp
    - cmvn.scp
    - text
    - utt2spk
  - The paths to `.ark` files in the `.scp` files should be valid paths.
  
Once you have this folder hierarchy ready, you can modify the input and output folder paths in the `scripts/prepare.sh` accordingly and launch the script. The script will create the uncompressed feature files in the format required by `nmtpytorch`. Specifically, the output folder hierarchy should look something like below:

```
/tmp/data/swbd/
├── eval2000_test
│   ├── feats_local.scp
│   └── segments.len
├── train_dev
│   ├── feats_local.scp
│   └── segments.len
└── train_nodup
    ├── feats_local.scp
    └── segments.len
```

