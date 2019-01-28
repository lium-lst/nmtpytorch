Automatic Speech Recognition (ASR)
--

Two example configuration files for character level and subword level
ASR systems. These experiments make use of the [ASR](https://github.com/lium-lst/nmtpytorch/blob/master/nmtpytorch/models/asr.py) model from `nmtpytorch`.

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
│   ├── feats_local.ark
│   ├── feats_local.scp
│   └── segments.len
├── train_dev
│   ├── feats_local.ark
│   ├── feats_local.scp
│   └── segments.len
└── train_nodup
    ├── feats_local.ark
    ├── feats_local.scp
    └── segments.len
```

**NOTE:** Unlike Kaldi, we remove the utterance ID columns from the label files for `nmtpytorch` so you need to make sure that the `text` files are in the **same order** with the `feats.scp` file.

Now if you look the provided configuration files, you will see that the speech modality tagged with `en_speech` keys are pointing towards the folders listed above:

```
[data]                                                     
root: /tmp/data/swbd                                       

train_set: {'en_speech': '${root}/train_nodup',            
            'en_text': '${root}/train_nodup/text.char.nmtpy'}                                                          

val_set: {'en_speech': '${root}/train_dev',                
          'en_text': '${root}/train_dev/text.char.nmtpy'}  

eval2000_set: {'en_speech': '${root}/eval2000_test'}       

[vocabulary]                                               
en_text: ${data:root}/train_nodup/text.char.vocab.nmtpy
```

### Adding label files and vocabularies

The last set of files to prepare are the target side transcript files tagged with `en_text` keys above. These are plain text files **without the utterance ID columns**. Each line corresponds to an utterance/segment and explicit spaces are defined with the `<s>` token. An example line should look like this:
```
y e a h <s> y e a h <s> w e l l <s> i - <s> i - <s> t h a t ' s <s> r i g h t <s> a n d <s> i t
```

**HINT:** You can use `scripts/word2char` to convert a word-level text file to the above format easily

On the other hand, a subword-level file prepared with `subword-nmt` tool looks like this:
```
all right th@@ an@@ ks bye bye
```

Once you have the transcript files preprocessed this way, you can run `nmtpy-build-vocab` to create the vocabulary file using the training sentence file:

```
$ nmtpy-build-vocab <preprocessed training set transcript file>
```

### Configuration Files

 - `asr-bilstmp-char.conf:` Character-level ASR baseline that uses character error rate (CER) as early-stopping metric.
 - `asr-bilstmp-s1k.conf:` BPE-level ASR baseline example. Here the early-stopping metric is WER. To correctly compute the WER over non-BPE files, a post-processing filter is activated in the configuration file: `eval_filters: de-bpe`

### Launching Training
See [this](https://github.com/lium-lst/nmtpytorch/wiki/Running-Experiments)

### Decoding Afterwards
Once training is over, you can use `nmtpy translate` command to decode arbitrary dev/test sets using beam search. For example to decode the `eval2000` set defined in the above config, you can run:

```
# batch_size: 32 beam_size:10 output file prefix: eval2000
# last argument is model checkpoint file
CUDA_VISIBLE_DEVICES=0 nmtpy translate -s eval2000 -b 32 -k 10 -o eval2000 <path to model .ckpt file>
```
