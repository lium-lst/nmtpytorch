# -*- coding: utf-8 -*-
import logging

import torch
from torch import nn

from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from ..utils.ml_metrics import Loss, Recall, Precision, F1
from ..utils.device import DEVICE
from ..utils.misc import pbar
from ..utils.data import sort_predictions
from ..datasets import MultimodalDataset
from ..metrics import Metric
from . import NMT

from ..layers import FF, Pool, ArgSelect

logger = logging.getLogger('nmtpytorch')


class LabelClassifier(NMT):
    supports_beam_search = False

    def set_defaults(self):
        self.defaults = {
            ############################
            ### Input feature projection
            ############################
            'feat_dim': 1024,           # Source feature dim
            'feat_proj_dim': 128,       # FF-layer dim
            'feat_proj_activ': None,    # Linear FF by default
            ##################################################################
            'enc_type': None,           # None, gru or lstm
            'enc_dim': 256,             # Hidden dim of the encoder
            'n_encoders': 1,            # Only used if enc_type != None
            'enc_dropout': 0.5,         # Only used if enc_type != None
            'enc_bidir': False,         # Bi-directional?
            ##################################################################
            'feat_aggregate': 'mean',   # How to pool feat matrix into vector?
            'weighted_loss': False,     # Weigh the labels with frequency
            'weight_type': 'custom',    #
            'threshold': 0.3,           # classification threshold
            'dropout': 0.5,             # Dropout p
            'short_list': 0,            # Reduce vocabulary if given
            'n_ff_layers': 1,           # How many stacked output layers?
            'ff_out_activ': 'tanh',     # If n_ff_layers > 1, use this non-lin
            ############################
            'direction': None,          # Network directionality, i.e. en->de
            'bucket_by': None,          # A key like 'en' to define w.r.t which dataset
                                        # the batches will be sorted
            'sampler_type': 'bucket',   # bucket or approximate
        }

    def __init__(self, opts):
        # Don't call NMT init as it's too different from this model
        nn.Module.__init__(self)

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
        self.src = self.topology.first_src
        self.trg = self.topology.first_trg

        # Load vocabularies here
        for name, fname in self.opts.vocabulary.items():
            self.vocabs[name] = Vocabulary(
                fname, name=name, short_list=self.opts.model['short_list'])

        # Set implicit label count for correct batch preparation
        self.topology[self.trg].kwargs['n_labels'] = len(self.vocabs[self.trg])

    def reset_parameters(self):
        for name, param in self.named_parameters():
            # Skip 1-d biases and scalars
            if param.requires_grad and param.dim() > 1:
                nn.init.kaiming_normal_(param.data)

    def setup(self, is_train=True):
        """Sets up NN topology by creating the layers."""
        feat_proj = FF(self.opts.model['feat_dim'],
                       self.opts.model['feat_proj_dim'],
                       activ=self.opts.model['feat_proj_activ'],
                       bias=False)

        layers = [feat_proj]
        out_dim = self.opts.model['feat_proj_dim']

        if self.opts.model['enc_type'] is not None:
            RNN = getattr(nn, self.opts.model['enc_type'].upper())
            enc = RNN(self.opts.model['feat_proj_dim'],
                      self.opts.model['enc_dim'],
                      self.opts.model['n_encoders'],
                      dropout=self.opts.model['enc_dropout'],
                      bidirectional=self.opts.model['enc_bidir'],
                      batch_first=False)
            out_dim = self.opts.model['enc_dim']
            layers.append(enc)
            layers.append(ArgSelect(0))
            if self.opts.model['enc_bidir']:
                layers.append(FF(out_dim * 2, out_dim, bias=False))

        # Pool states over the 1st dim (sequential)
        layers.append(
            Pool(self.opts.model['feat_aggregate'], pool_dim=0, keepdim=False))

        # Additional layers
        if self.opts.model['n_ff_layers'] > 1:
            for i in range(self.opts.model['n_ff_layers'] - 1):
                layers.append(
                    FF(out_dim, out_dim, activ=self.opts.model['ff_out_activ']))

        # Output layer
        layers.append(
            FF(out_dim, len(self.vocabs[self.trg]), bias=False))

        if self.opts.model['dropout'] > 0:
            layers.append(nn.Dropout(self.opts.model['dropout']))

        # Construct one encoder abstraction
        self.encoder = nn.Sequential(*layers)

        # Loss manipulation
        counts = None
        self.eval_loss = nn.MultiLabelSoftMarginLoss(reduction='sum')
        if self.opts.model['weighted_loss']:
            counts = torch.FloatTensor(
                list(self.vocabs[self.trg].counts.values()))
            if self.opts.model['weight_type'] == 'custom':
                counts.div_(counts.max()).sub_(1).mul_(-10).add_(1)
            elif self.opts.model['weight_type'] == 'inv_log':
                counts.log_().round_()
                counts = 1 / (counts.div(counts.min()))
            self.train_loss = nn.MultiLabelSoftMarginLoss(
                weight=counts, reduction='sum')
        else:
            self.train_loss = self.eval_loss

        if self.opts.model['threshold'] > 0:
            self.thresholder = lambda x: x.gt(self.opts.model['threshold']).float()
        else:
            self.thresholder = lambda x: x.round()

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        dataset = MultimodalDataset(
            data=self.opts.data['{}_set'.format(split)],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            max_len=None,
            bucket_by=self.opts.model['bucket_by'],
            bucket_order=None,
            sampler_type=self.opts.model['sampler_type'])
        logger.info(dataset)
        return dataset

    def forward(self, batch, **kwargs):
        """Computes the forward-pass of the network and returns batch loss.

        Arguments:
            batch (dict): A batch of samples with keys designating the source
                and target modalities.

        Returns:
            Tensor:
                A scalar loss normalized w.r.t batch size and token counts.
        """
        # Encode
        out = self.encoder(batch[self.src])

        # Prepare results
        results = {'loss': 0, 'n_items': out.shape[0]}

        # Loss can only be computed if target labels available
        loss_fn = self.train_loss if self.training else self.eval_loss
        if self.trg in batch:
            labels = batch[self.trg].squeeze_(0)
            results['loss'] = loss_fn(out, labels)

        # Prepare predictions for further metric computations
        results['preds'] = out.detach().sigmoid()

        return results

    def test_performance(self, data_loader, dump_file=None):
        """Computes test set loss over the given DataLoader instance."""
        loss = Loss()
        rec = Recall()
        prec = Precision()
        f1 = F1()

        if dump_file:
            results = []

        for batch in pbar(data_loader, unit='batch'):
            batch.device(DEVICE)
            out = self.forward(batch)

            if dump_file:
                results.append(out['preds'].cpu().numpy())

            if self.trg in batch:
                labels = batch[self.trg]
                loss.update(out['loss'], out['n_items'])
                # Apply classifier threshold
                preds = self.thresholder(out['preds'])
                rec.update(preds, labels)
                prec.update(preds, labels)
                f1.update(preds, labels)

        if dump_file:
            import numpy as np
            results = sort_predictions(data_loader, np.concatenate(results))
            sample_idxs = data_loader.dataset.datasets[self.src].keys
            rdict = {}

            with open('{}.labels.txt'.format(dump_file), 'w') as f:
                for idx, scores in zip(sample_idxs, results):
                    rdict[idx] = scores.reshape(1, -1)

                    # For inspection purposes, dump only top 100 labels
                    label_str = self.vocabs[self.trg].idxs_to_sent(
                        scores.argsort()[::-1][:100], debug=True)
                    f.write('{}\t{}\n'.format(idx, label_str))

                # Dump raw score vectors as .npz
                np.savez_compressed(dump_file, **rdict)

        if data_loader.dataset.n_targets > 0:
            return [
                Metric('LOSS', loss.get(), higher_better=False),
                rec.compute(),
                prec.compute(),
                f1.compute(),
            ]
        else:
            return []
