# -*- coding: utf-8 -*-
import logging

import torch
from torch import nn
from torch import distributions

from ..layers import FF, ArgSelect
from ..layers.decoders import get_decoder
from ..layers import ArgSelect
from ..utils.misc import get_n_params
from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from ..utils.ml_metrics import Loss
from ..utils.device import DEVICE
from ..utils.misc import pbar
from ..datasets import MultimodalDataset
from ..metrics import Metric

from . import NMT

logger = logging.getLogger('nmtpytorch')


class Rationale(NMT):
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            'emb_dim': 128,             # Source and target embedding sizes
            'gen_dim': 256,             # Encoder hidden size
            'gen_type': 'gru',          # Encoder type (gru|lstm|bag|cnn?)
            'gen_lnorm': False,         # Add layer-normalization to encoder output
            'gen_bidir': True,          # Bi-directional encoder
            'n_encoders': 1,            # Number of stacked encoders
            'dec_dim': 256,             # Decoder hidden size
            'dec_type': 'gru',          # Decoder type (gru|lstm)
            'dec_variant': 'cond',      # (cond|simplegru|vector)
            'dec_init': 'mean_ctx',     # How to initialize decoder (zero/mean_ctx/feats)
            'dropout': 0,               # Simple dropout to source embeddings
            'tied_emb': False,          # Share embeddings: (False|2way|3way)
            'direction': None,          # Network directionality, i.e. en->de
            'max_len': 80,              # Reject sentences where 'bucket_by' length > 80
            'bucket_by': None,          # A key like 'en' to define w.r.t which dataset
                                        # the batches will be sorted
            'bucket_order': None,       # Curriculum: ascending/descending/None
            'sampler_type': 'bucket',   # bucket or approximate
            'out_logic': 'simple',      # 'simple' or 'deep' output
                                        #
            'lambda_coherence': 0,      # Coherence penalty
            'lambda_sparsity': 0,       # Sparsity penalty
            'sample': 1,                # How many samples to get
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
            #self.trg_emb.weight.data[0].fill_(0)

    def setup(self, is_train=True):
        """Sets up NN topology by creating the layers."""
        # Source embeddings
        self.src_emb = nn.Embedding(
            self.n_src_vocab, self.opts.model['emb_dim'], padding_idx=0)

        ##################
        # Generator block
        ##################
        gen_layers = []
        self.ctx_size = self.opts.model['gen_dim']
        if self.opts.model['gen_type'] in ('lstm', 'gru'):
            if self.opts.model['gen_bidir']:
                self.ctx_size *= 2

            RNN = getattr(nn, self.opts.model['gen_type'].upper())
            dropout = self.opts.model['dropout']
            if self.opts.model['n_encoders'] == 1:
                dropout = 0

            # RNN Encoder
            gen_layers.append(RNN(
                self.opts.model['emb_dim'], self.opts.model['gen_dim'],
                self.opts.model['n_encoders'], batch_first=False,
                dropout=dropout,
                bidirectional=self.opts.model['gen_bidir']))
            gen_layers.append(ArgSelect(0))
        else:
            raise NotImplementedError('gen_type unknown')

        # Probability of selection
        gen_layers.append(FF(self.ctx_size, 1, activ=None))

#        if self.opts.model['gen_lnorm']:
            ## Add layer_norm
            #gen_layers.append(nn.LayerNorm(self.ctx_size))
        #if self.opts.model['gen_proj']:
            ## Add a projection layer
            #gen_layers.append(FF(self.ctx_size, self.ctx_size, activ='tanh'))
#        if self.opts.model['dropout'] > 0:
            #gen_layers.append(nn.Dropout(p=self.opts.model['dropout']))

        # Create the encoder wrapper
        self.gen = nn.Sequential(*gen_layers)

        ################
        # Create Decoder
        ################
#        Decoder = get_decoder(self.opts.model['dec_variant'])
        #self.dec = Decoder(
            #input_size=self.opts.model['emb_dim'],
            #hidden_size=self.opts.model['dec_dim'],
            #n_vocab=self.n_trg_vocab,
            #rnn_type=self.opts.model['dec_type'],
            #ctx_size_dict=self.ctx_sizes,
            #ctx_name=str(self.sl),
            #tied_emb=self.opts.model['tied_emb'],
            #dec_init=self.opts.model['dec_init'],
            #att_bottleneck='hid',
            #dropout_out=self.opts.model['dropout'],
            #out_logic=self.opts.model['out_logic'])

        ## Share encoder and decoder weights
        #if self.opts.model['tied_emb'] == '3way':
            #self.enc.emb.weight = self.dec.emb.weight

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        self.dataset = MultimodalDataset(
            data=self.opts.data['{}_set'.format(split)],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model['max_len'],
            bucket_order=self.opts.model['bucket_order'],
            sampler_type=self.opts.model['sampler_type'])
        logger.info(self.dataset)
        return self.dataset

    def gumbel(self, x, temp):
        noise = torch.rand(x.size())
        noise.add_(1e-9).log_().neg_()
        noise.add_(1e-9).log_().neg_()
        y = (x + noise) / temp
        yy = torch.softmax(y.view(-1, x.size()[-1]), dim=-1)
        return yy.view_as(x)

    def generate(self, idxs, **kwargs):
        # embs: T x B x D
        embs = self.src_emb(idxs)
        # p_z: B x T (logits)
        p_z = self.gen(embs).squeeze(-1).t()

        dist = distributions.Bernoulli(logits=p_z)
        # Get N samples -> N x B x T
        z = dist.sample((self.opts.model['sample'], ))

        raise Exception()

        return p_z

    def forward(self, batch, **kwargs):
        p_z = self.generate(batch[self.sl])

        #result = self.dec(self.encode(batch), batch[self.tl])
        #result['n_items'] = torch.nonzero(batch[self.tl][1:]).shape[0]
        return result
