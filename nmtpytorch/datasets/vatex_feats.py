# -*- coding: utf-8 -*-
from . import KeyedNPZDataset


class VATEXFeaturesDataset(KeyedNPZDataset):
    def __init__(self, fname, **kwargs):
        super().__init__(fname, **kwargs)
        self.repeat_by = kwargs.get('repeat_by', 1)

        if self.repeat_by > 1:
            # Every feature represents many captions/sentences
            self.size *= self.repeat_by
            copy_keys = list(self.keys)
            self.keys = [k for k in copy_keys for i in range(self.repeat_by)]
            if self.lengths is not None:
                copy_lens = self.lengths[:]
                self.lengths = [k for k in copy_lens for i in range(self.repeat_by)]
