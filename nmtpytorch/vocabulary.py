# -*- coding: utf-8 -*-

import json
import pathlib
from collections import OrderedDict

from .utils.misc import get_language


class Vocabulary(object):
    """Smart vocabulary class for integer<->token mapping."""

    TOKENS = {"<pad>": 0,
              "<bos>": 1,
              "<eos>": 2,
              "<unk>": 3}

    def __init__(self, vocab, lang=None):
        self.vocab = str(pathlib.Path(vocab).expanduser())
        self.lang = lang
        self._map = None
        self._imap = None
        self._allmap = None
        self.n_tokens = None

        if self.lang is None:
            self.lang = get_language(self.vocab)

        self._map = json.load(open(self.vocab))

        # Sanity check for placeholder tokens
        for tok, idx in self.TOKENS.items():
            assert self._map.get(tok, -1) == idx, \
                "%s not found in vocabulary." % tok

        # Set # of tokens
        self.n_tokens = len(self._map)

        # Invert dictionary
        self._imap = OrderedDict([(v, k) for k, v in self._map.items()])

        # Merge forward and backward lookups into single dict for convenience
        self._allmap = OrderedDict()
        self._allmap.update(self._map)
        self._allmap.update(self._imap)

        assert len(self._allmap) == (len(self._map) + len(self._imap)), \
            "Merged vocabulary size is not equal to sum of both."

    def __getitem__(self, key):
        return self._allmap[key]

    def __len__(self):
        return len(self._map)

    def sent_to_idxs(self, line, limit=0, explicit_bos=False):
        """Convert from list of strings to list of token indices."""
        tidxs = []

        if explicit_bos:
            tidxs.append(self.TOKENS["<bos>"])

        for tok in line.split():
            tidxs.append(self._map.get(tok, self.TOKENS["<unk>"]))

        if limit > 0:
            tidxs = [tidx if tidx < limit else self.TOKENS["<unk>"]
                     for tidx in tidxs]

        # Append explicit <eos>
        tidxs.append(self.TOKENS["<eos>"])

        return tidxs

    def idxs_to_sent(self, tidxs, do_join=True):
        """Convert from token indices to string representation."""
        result = []
        for tidx in tidxs:
            if tidx == self.TOKENS["<eos>"]:
                break
            result.append(self._imap.get(tidx, self.TOKENS["<unk>"]))

        if do_join:
            return " ".join(result)
        else:
            return result

    def __repr__(self):
        return "Vocabulary of %d items (lang=%s)" % (self.n_tokens,
                                                     self.lang)


if __name__ == '__main__':
    vocab_file = "data/bpe/train.norm.tok.lc.bpe10000.vocab.en"
    v = Vocabulary(vocab_file, lang='en')

    sent = 'he is a good man .'
    idxs = v.sent_to_idxs(sent)
    assert v.idxs_to_sent(idxs) == sent, "Test 1 failed."
