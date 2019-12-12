# -*- coding: utf-8 -*-
import logging

import torch
from torch import nn

from ..layers.transformers import *

from . import NMT

logger = logging.getLogger('nmtpytorch')


class TransformerNMT(NMT):
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            'model_dim': 512,           # model_dim
            'ff_dim': 2048,             # Positionwise FF inner dimension
            'n_enc_layers': 6,          # Number of encoder layers
            'n_dec_layers': 6,          # Number of decoder layers
            'n_heads': 8,               # Number of attention heads
            'direction': None,          # Network directionality, i.e. en->de
            'max_len': None,            # Reject sentences where 'bucket_by' length > 80
            'bucket_by': None,          # A key like 'en' to define w.r.t which dataset
                                        # the batches will be sorted
            'bucket_order': None,       # Curriculum: ascending/descending/None
            'sampler_type': 'bucket',   # bucket or approximate
            'short_list': 0,            # Vocabulary short listing
        }

    def __init__(self, opts):
        super().__init__(opts)

    def reset_parameters(self):
        for name, param in self.named_parameters():
            # Skip 1-d biases and scalars
            if param.requires_grad and param.dim() > 1:
                nn.init.kaiming_normal_(param.data)
        # Reset padding embedding to 0
        with torch.no_grad():
            self.src_emb.weight.data[0].fill_(0)
            self.trg_emb.weight.data[0].fill_(0)

    def setup(self, is_train=True):
        """Sets up NN topology by creating the layers."""
        # Create the embeddings
        self.src_emb = TFEmbedding(self.n_src_vocab, self.opts.model['model_dim'])
        self.trg_emb = TFEmbedding(self.n_trg_vocab, self.opts.model['model_dim'])
        self.enc = TFEncoder(
            self.opts.model['model_dim'], self.opts.model['ff_dim'],
            self.opts.model['n_heads'], self.opts.model['n_enc_layers'])
        self.dec = TFDecoder(
            self.opts.model['model_dim'], self.opts.model['ff_dim'],
            self.opts.model['n_heads'], self.opts.model['n_dec_layers'])
        self.seq_loss = torch.nn.NLLLoss(reduction='sum', ignore_index=0)

    def encode(self, batch, **kwargs):
        # mask: (tstep, bsize)
        mask = batch[self.sl].ne(0).float()

        # embs: (tstep, bsize, dim)
        embs = self.src_emb(batch[self.sl])
        h, mask = self.enc(embs, mask=mask)

        d = {str(self.sl): (h, mask)}
        return d

    def forward(self, batch, **kwargs):
        # Get loss dict
        enc = self.encode(batch)

        dec_input = batch[self.tl]

#         result = self.dec(self.encode(batch), batch[self.tl])
        # result['n_items'] = torch.nonzero(batch[self.tl][1:]).shape[0]
        # return result
