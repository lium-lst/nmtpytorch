# -*- coding: utf-8 -*-
import logging

from ..datasets import MultimodalDataset
from .nmt import NMT

logger = logging.getLogger('nmtpytorch')


class MNMTDecinit(NMT):
    """An end-to-end sequence-to-sequence NMT model with auxiliary visual
    features used for decoder initialization. See WMT17 LIUM-CVC paper.
    We only change the dataset and the encode() for this model.
    """
    def __init__(self, opts):
        super().__init__(opts)

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        dataset = MultimodalDataset(
            data=self.opts.data[split + '_set'],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model.get('max_len', None))
        logger.info(dataset)
        return dataset

    def encode(self, batch, **kwargs):
        return {
            'feats': (batch['feats'], None),
            str(self.sl): self.enc(batch[self.sl]),
        }
