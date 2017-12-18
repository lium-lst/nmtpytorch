# -*- coding: utf-8 -*-
import re
from pathlib import Path

from .misc import get_temp_file, fopen


class FilterChain(object):
    filters = {
        'de-bpe': lambda s: s.replace("@@ ", "").replace("@@", ""),
        # Converts segmentations of <tag:morpheme> to normal form
        'de-segment': lambda s: re.sub(' *<.*?:(.*?)>', '\\1', s),
        # Space delim character sequence to non-tokenized normal word form
        'c2w': lambda s: s.replace(' ', '').replace('<s>', ' ').strip(),
        # Filters out fillers from compound splitted sentences
        'de-compound': lambda s: (s.replace(" @@ ", "").replace(" @@", "")
                                   .replace(" @", "").replace("@ ", "")),
        'lower': lambda s: s.lower(),
        'upper': lambda s: s.upper(),
    }

    def __init__(self, _filters):
        self._filters = _filters.split(',')
        assert not set(self._filters).difference(self.filters.keys()), \
            "Unknown evaluation filter given in train.eval_filters"
        self.funcs = [self.filters[k] for k in self._filters]

    def _apply(self, list_of_strs):
        for func in self.funcs:
            list_of_strs = [func(s) for s in list_of_strs]
        return list_of_strs

    def __call__(self, inp):
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
