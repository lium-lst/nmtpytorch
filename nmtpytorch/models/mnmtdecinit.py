# -*- coding: utf-8 -*-
import logging

from ..datasets import Multi30kDataset
from .nmt import NMT

logger = logging.getLogger('nmtpytorch')


class MNMTDecinit(NMT):
    """An end-to-end sequence-to-sequence NMT model with auxiliary visual
    features used for decoder initialization. See WMT17 LIUM-CVC paper.
    We only change the dataset and the encode() for MNMTFeatures model.
    """
    def __init__(self, opts):
        super().__init__(opts)
        # This is not used for Multi30kDataset
        self.opts.model.pop('max_trg_len')

    def load_data(self, split):
        """Loads the requested dataset split."""
        self.datasets[split] = Multi30kDataset(
            data_dict=self.opts.data[split + '_set'],
            vocabs=self.vocabs,
            topology=self.topology)
        logger.info(self.datasets[split])

    def encode(self, batch):
        return {
            'feats': (batch['feats'], None),
            str(self.sl): self.enc(batch[self.sl]),
        }
