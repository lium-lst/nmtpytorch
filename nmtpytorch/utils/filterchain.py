# -*- coding: utf-8 -*-
import re
from pathlib import Path

from .misc import get_temp_file, fopen


class FilterChain:
    """A sequential filter chain to post-process list of tokens.

        Arguments:
            filters(str): A string containing comma-separated list of filters
                to apply.

        Available Filters:
            'de-bpe': Stitches back subword units produced with apply_bpe
            'de-spm': Stitches back sentence pieces produced with spm_encode
            'de-segment': Converts <tag:morpheme> to normal form
            'de-compond': Stitches back German compound splittings
            'c2w': Stitches back space delimited characters to words.
                Necessary for word-level BLEU, etc. when using CharNMT.
            'lower': Lowercase.
            'upper': Uppercase.
            'de-hyphen': De-hyphenate 'foo @-@ bar' constructs of Moses.
    """
    FILTERS = {
        'de-bpe': lambda s: s.replace("@@ ", "").replace("@@", ""),
        'de-tag': lambda s: re.sub('<[a-zA-Z][a-zA-Z]>', '', s),
        # Decoder for Google sentenpiece
        # only for default params of spm_encode
        'de-spm': lambda s: s.replace(" ", "").replace("\u2581", " ").strip(),
        # Converts segmentations of <tag:morpheme> to normal form
        'de-segment': lambda s: re.sub(' *<.*?:(.*?)>', '\\1', s),
        # Space delim character sequence to non-tokenized normal word form
        'c2w': lambda s: s.replace(' ', '').replace('<s>', ' ').strip(),
        # Filters out fillers from compound splitted sentences
        'de-compound': lambda s: (s.replace(" @@ ", "").replace(" @@", "")
                                  .replace(" @", "").replace("@ ", "")),
        # de-hyphenate when -a given to Moses tokenizer
        'de-hyphen': lambda s: re.sub('\s*@-@\s*', '-', s),
        'lower': lambda s: s.lower(),
        'upper': lambda s: s.upper(),
    }

    def __init__(self, filters):
        self.filters = filters.split(',')
        assert not set(self.FILTERS).difference(self.FILTERS.keys()), \
            "Unknown evaluation filter given in train.evalfilters"
        self.funcs = [self.FILTERS[k] for k in self.filters]

    def _apply(self, list_of_strs):
        for func in self.funcs:
            list_of_strs = [func(s) for s in list_of_strs]
        return list_of_strs

    def __call__(self, inp):
        """Applies the filterchain on a given input.

        Arguments:
            inp(pathlib.Path or list): If a `Path` given, temporary
                file(s) with filters applied are returned. The `Path` can
                also be a glob expression.
                Otherwise, a list with filtered sentences is returned.
        """
        if isinstance(inp, Path):
            # Need to create copies of reference files with filters applied
            # and return their paths instead
            fnames = inp.parent.glob(inp.name)
            new_fnames = []
            for fname in fnames:
                lines = []
                f = fopen(fname)
                for line in f:
                    lines.append(line.strip())
                f.close()
                f = get_temp_file()
                for line in self._apply(lines):
                    f.write(line + '\n')
                f.close()
                new_fnames.append(f.name)
            return new_fnames

        elif isinstance(inp, list):
            return self._apply(inp)

    def __repr__(self):
        return "FilterChain({})".format(" -> ".join(self.filters))
