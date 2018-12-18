Examples
--

Here you can find example configuration files that are tied to specific versions
of nmtpytorch. You need to set paths accordingly in order for the configurations
to work correctly.

## Multimodal task (En->Fr)

### mmt-task-en-fr-nmt.conf

A baseline NMT for En->Fr language pair
of Multi30K. You can download the Multi30K dataset from [here](https://github.com/multi30k/dataset).

The dataset files are suffixed with `lc.norm.tok` in this experiment which
means that Moses scripts were used to lowercase -> normalize-punctuation -> tokenize
the corpora. Specifically for tokenization, we enable `-a` option to aggressively
split the hyphens.

Next you need to run `nmtpy-build-vocab` on the `train.lc.norm.tok.*` files
to construct the vocabularies. You should now be able to train the systems
accordingly.

### mmt-task-en-fr-encdecinit.conf

A baseline multimodal NMT for En->Fr language pair of Multi30K. You need
to have `.npy` feature files for image features in order to train this model.

A feature file should contain a tensor of shape `(n, feat_dim)` where `n` is the
number of sentences of the split and `feat_dim` is the dimensionality for the features.

Depending on `d`, you need to adjust the `feat_dim` option in the configuration file.

You can download the provided ResNet-50 feature files for the shared task in WMT18
from [here](https://drive.google.com/drive/folders/1I2ufg3rTva3qeBkEc-xDpkESsGkYXgCf?usp=sharing).
