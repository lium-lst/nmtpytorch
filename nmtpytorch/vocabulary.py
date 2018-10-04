# -*- coding: utf-8 -*-
import json
import pathlib
import logging
from collections import OrderedDict

logger = logging.getLogger('nmtpytorch')


class Vocabulary:
    """Smart vocabulary class for integer<->token mapping."""

    TOKENS = {"<pad>": 0,
              "<bos>": 1,
              "<eos>": 2,
              "<unk>": 3}

    def __init__(self, vocab, name):
        self.vocab = pathlib.Path(vocab).expanduser()
        self.name = name
        self._map = None
        self._imap = None
        self.freqs = None
        self.counts = None
        self._allmap = None
        self.n_tokens = None

        # Load file
        data = json.load(open(self.vocab))

        # Detect vocabulary: values can be int or a string of type "id count"
        elem = next(iter(data.values()))
        if isinstance(elem, str):
            self._map = {k: int(v.split()[0]) for k, v in data.items()}
            self.counts = {k: int(v.split()[1]) for k, v in data.items()}
            total_count = sum(self.counts.values())
            self.freqs = {k: v / total_count for k, v in self.counts.items()}
        elif isinstance(elem, int):
            self._map = data
        else:
            raise RuntimeError('Unknown vocabulary format.')

        # Sanity check for placeholder tokens
        for tok, idx in self.TOKENS.items():
            if self._map.get(tok, -1) != idx:
                logger.info('{} not found in {}'.format(tok, self.vocab.name))

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

    def sent_to_idxs(self, line, explicit_bos=False, explicit_eos=True):
        """Convert from list of strings to list of token indices."""
        tidxs = []

        if explicit_bos:
            tidxs.append(self.TOKENS["<bos>"])

        for tok in line.split():
            tidxs.append(self._map.get(tok, self.TOKENS["<unk>"]))

        if explicit_eos:
            # Append <eos>
            tidxs.append(self.TOKENS["<eos>"])

        return tidxs

    def idxs_to_sent(self, idxs, debug=False):
        """Convert integer hypothesis to string."""
        result = []
        for idx in idxs:
            if not debug and idx == self.TOKENS["<eos>"]:
                break
            result.append(self._imap.get(idx, self.TOKENS["<unk>"]))

        return " ".join(result)

    def list_of_idxs_to_sents(self, lidxs):
        """Convert a list of integer hypotheses to list of strings."""
        results = []
        unk = self.TOKENS["<unk>"]
        for idxs in lidxs:
            result = []
            for idx in idxs:
                if idx == self.TOKENS["<eos>"]:
                    break
                result.append(self._imap.get(idx, unk))
            results.append(" ".join(result))
        return results

    def __repr__(self):
        return "Vocabulary of %d items (name=%s)" % (self.n_tokens,
                                                     self.name)
