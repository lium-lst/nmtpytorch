# -*- coding: utf-8 -*-
import logging

import torch
from torch import nn

from ..layers import FF, get_partial_embedding_layer
from ..utils.misc import get_n_params
from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from ..utils.ml_metrics import Loss
from ..utils.device import DEVICE
from ..utils.misc import pbar
from ..datasets import MultimodalDataset
from ..metrics import Metric

logger = logging.getLogger('nmtpytorch')


class NLI(nn.Module):
    """A very simple BiLSTM NLI baseline."""
    supports_beam_search = False

    def set_defaults(self):
        self.defaults = {
            'bidirectional': True,      # Bi-directional LSTM
            'emb_dim': 300,             # Input embedding size
            'inp_dim': 300,             # Projected embedding size
            'enc_dim': 300,             # Encoder hidden size
            'proj_dim': 600,            # Output proj dims
            'nonlin': 'tanh',           # Non-linearity type
            'emb_zero_oov': False,      # Init to zero the OOV words
            'n_encoders': 1,            # Number of stacked encoders
            'dropout': 0,               # Global dropout value
            'direction': None,          # Network directionality, i.e. en->de
            'max_len': None,            # Reject sentences where 'bucket_by' length > 80
            'bucket_by': None,          # A key like 'en' to define w.r.t which dataset
                                        # the batches will be sorted
            'bucket_order': None,       # Curriculum: ascending/descending/None
            'sampler_type': 'bucket',   # bucket or approximate
            'init_emb': None,           # Pretrained .pkl file which is a dict
            'init_emb_freeze': 'none',  # none/all/partial
        }

    def __init__(self, opts):
        super().__init__()

        # opts -> config file sections {.model, .data, .vocabulary, .train}
        self.opts = opts

        # Vocabulary objects
        self.vocabs = {}

        # Each auxiliary loss should be stored inside this dictionary
        # in order to be taken into account by the mainloop for multi-tasking
        self.aux_loss = {}

        # Setup options
        self.opts.model = self.set_model_options(opts.model)

        # Parse topology & languages
        self.topology = Topology(self.opts.model['direction'])

        # Load vocabularies here
        for name, fname in self.opts.vocabulary.items():
            self.vocabs[name] = Vocabulary(fname, name=name)

    def __repr__(self):
        s = super().__repr__() + '\n'
        for vocab in self.vocabs.values():
            s += "{}\n".format(vocab)
        s += "{}\n".format(get_n_params(self))
        return s

    def set_model_options(self, model_opts):
        self.set_defaults()
        for opt, value in model_opts.items():
            if opt in self.defaults:
                # Override defaults from config
                self.defaults[opt] = value
            else:
                logger.info('Warning: unused model option: {}'.format(opt))
        return self.defaults

    def reset_parameters(self):
        return
        for name, param in self.named_parameters():
            if name == 'emb.weight':
                # NOTE: Do not reinit embeddings for now
                continue
            # Skip 1-d biases and scalars
            if param.requires_grad and param.dim() > 1:
                nn.init.kaiming_normal_(param.data)

    def setup(self, is_train=True):
        """Sets up NN topology by creating the layers."""
        ########################
        # Create Textual Encoder
        ########################
        self.lstm = nn.LSTM(
            input_size=self.opts.model['inp_dim'],
            hidden_size=self.opts.model['enc_dim'],
            num_layers=self.opts.model['n_encoders'],
            bidirectional=self.opts.model['bidirectional'])

        if self.opts.model['init_emb']:
            # Use pretrained embeddings
            # NOTE: Fix grad_mask thing
            self.emb = get_partial_embedding_layer(
                self.vocabs['pre'], self.opts.model['emb_dim'],
                pretrained_file=self.opts.model['init_emb'],
                freeze=self.opts.model['init_emb_freeze'],
                oov_zero=self.opts.model['emb_zero_oov'])
        else:
            # Train embedding from scratch
            self.emb = nn.Embedding(len(self.vocabs['pre']),
                                    self.opts.model['emb_dim'],
                                    padding_idx=0)

        self.emb_proj = FF(
            self.opts.model['emb_dim'], self.opts.model['inp_dim'],
            activ=self.opts.model['nonlin'])

        inp_dim = self.opts.model['enc_dim'] * 2
        proj_dim = self.opts.model['proj_dim']
        if self.opts.model['bidirectional']:
            inp_dim *= 2

        self.output = nn.Sequential(
            FF(inp_dim, proj_dim, activ=self.opts.model['nonlin']),
            nn.Dropout(self.opts.model['dropout']),
            FF(proj_dim, proj_dim, activ=self.opts.model['nonlin']),
            nn.Dropout(self.opts.model['dropout']),
            FF(proj_dim, proj_dim//2, activ=self.opts.model['nonlin']),
            nn.Dropout(self.opts.model['dropout']),
            FF(proj_dim//2, self.vocabs['lb'].n_tokens, bias=False),
            nn.LogSoftmax(dim=-1))
        self.loss = nn.NLLLoss(reduction='sum')

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        dataset = MultimodalDataset(
            data=self.opts.data['{}_set'.format(split)],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model['max_len'],
            bucket_order=self.opts.model['bucket_order'],
            sampler_type=self.opts.model['sampler_type'])
        logger.info(dataset)
        return dataset

    def encode(self, batch, **kwargs):
        pre, _ = self.lstm(self.emb_proj(self.emb(batch['pre'])))
        hyp, _ = self.lstm(self.emb_proj(self.emb(batch['hyp'])))

        pre_mask = (batch['pre'] > 0).long().sum(0).sub(1)
        hyp_mask = (batch['hyp'] > 0).long().sum(0).sub(1)
        last_pre = pre[pre_mask, range(pre.shape[1])]
        last_hyp = hyp[hyp_mask, range(hyp.shape[1])]

        return torch.cat((last_pre, last_hyp), dim=-1)

    def forward(self, batch, **kwargs):
        probs = self.output(self.encode(batch))
        loss = self.loss(probs, batch['lb'].squeeze(0))
        d = {
            'loss': loss,
            'probs': probs,
            'n_items': batch.size,
        }
        return d

    def test_performance(self, data_loader, dump_file=None):
        """Computes test set loss over the given DataLoader instance."""
        loss = Loss()
        acc = Loss()

        for batch in pbar(data_loader, unit='batch'):
            batch.device(DEVICE)
            out = self.forward(batch)
            loss.update(out['loss'], out['n_items'])
            acc.update(
                out['probs'].max(1)[1].eq(batch['lb'].squeeze()).float().sum(),
                out['n_items'])

        return [
            Metric('LOSS', loss.get(), higher_better=False),
            Metric('ACC', acc.get(), higher_better=True),
        ]
