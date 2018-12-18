Examples
--

Here you can find example configuration files that are tied to specific versions
of nmtpytorch. You need to set paths accordingly in order for the configurations
to work correctly.

### Multimodal task (En->Fr)

`mmt-task-en-fr-nmt.conf` provides a baseline NMT for En->Fr language pair
of Multi30K. You can download the Multi30K dataset from [here](https://github.com/multi30k/dataset).

The dataset files are suffixed with `lc.norm.tok` in this experiment which
means that Moses scripts were used to lowercase -> normalize-punctuation -> tokenize
the corpora. Specifically for tokenization, we enable `-a` option to aggressively
split the hyphens.

Next you need to run `nmtpy-build-vocab` on the `train.lc.norm.tok.*` files
to construct the vocabularies. You should now be able to train the systems
accordingly.
