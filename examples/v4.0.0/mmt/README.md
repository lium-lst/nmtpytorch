Examples
--

Here you can find example configuration files that are tied to specific versions
of nmtpytorch. You need to set paths accordingly in order for the configurations
to work correctly.

## Multimodal task (En->Fr)

**NOTE:** These examples do not use BPE-segmented files, instead they simply
use word forms.

The dataset files are suffixed with `lc.norm.tok` in this experiment which
means that Moses scripts were used to lowercase -> normalize-punctuation -> tokenize
the corpora. Specifically for tokenization, we enable `-a` option to aggressively
split the hyphens. The following pipeline should do the trick (moses scripts should
be in `$PATH` for the following to work as-is):

```bash
for split in train val test_201*flickr; do
  for llang in en fr; do
    lowercase.perl < ${split}.${llang} | normalize-punctuation.perl -l $llang | \
      tokenizer.perl -q -a -l $llang -threads 4 > ${split}.lc.norm.tok.${llang}
  done
done
```

Next you need to run `nmtpy-build-vocab` on the `train.lc.norm.tok.*` files
to construct the vocabularies. You should now be able to train the systems
accordingly.

**NOTE:** For multimodal systems, you may want to L2-normalize the feature files
and save the normalized versions, see [WMT18 paper for LIUM-CVC](https://arxiv.org/abs/1809.00151):

```python
x = np.load('foo.npy')
np.save('foo-l2norm.npy', x / np.linalg.norm(x, axis=-1, keepdims=True))
```

### mmt-task-en-fr-nmt.conf

A baseline NMT for En->Fr language pair
of Multi30K. You can download the Multi30K dataset from [here](https://github.com/multi30k/dataset).

### mmt-task-en-fr-encdecinit.conf

- A baseline multimodal NMT for En->Fr language pair of Multi30K. You need
  to have `.npy` feature files for image features in order to train this model.

- A feature file should contain a tensor of shape `(n, feat_dim)` where `n` is the
  number of sentences of the split and `feat_dim` is the dimensionality for the features.

- Depending on `feat_dim`, you need to adjust the `feat_dim` option in the configuration file.

- You can download the provided ResNet-50 feature files for the WMT18 shared task
  from [here](https://drive.google.com/drive/folders/1I2ufg3rTva3qeBkEc-xDpkESsGkYXgCf?usp=sharing).

- The feature files for this model have `avgpool` in their filenames and the
  `feat_dim` is `2048`.

### mmt-task-en-fr-multimodalatt.conf

A multimodal attentive NMT baseline replicating [this paper](https://arxiv.org/abs/1609.03976).
You now need to use the convolutional feature files that can be downloaded from the same link above.

- The feature files for this model have `res4frelu` in their filenames and the `feat_dim` is `1024`.

#### More variants

 - You can switch to [hierarchical attention](https://arxiv.org/pdf/1704.06567.pdf) by
   changing `fusion_type: concat` to `fusion_type: hierarchical` in the `*multimodalatt.conf`
   file.
