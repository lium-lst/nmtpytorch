# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch
import torch.nn as nn

from ..layers import BiLSTMp, ConditionalDecoder
from ..utils.misc import get_n_params
from ..utils.nn import ModuleDict
from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from ..utils.ml_metrics import Loss
from ..datasets import MultimodalDataset
from ..metrics import Metric

logger = logging.getLogger('nmtpytorch')


# A multi-task training regime to achieve ASR and SLT by sharing a
# speech encoder.


class SLTASROneToMany(nn.Module):
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            'sp_feat_dim': 43,               # Speech features dimensionality
            'sp_enc_dim': 256,               # Speech encoder dim
            'sp_enc_layers': '1_1_2_2_1_1',  # layer configuration
            'sp_proj_dim': 300,              # Intra-LSTM projection layer
            'sp_proj_activ': 'tanh',         # Intra-LSTM projection activation
            'slt_emb_dim': 128,              # SLT Decoder embedding dim
            'slt_dec_dim': 256,              # SLT Decoder hidden size
            'slt_tied_dec_embs': False,      # SLT Share decoder embeddings
            'slt_att_temp': 1.0,             # SLT attention temperature
            'asr_emb_dim': 128,              # ASR Decoder embedding dim
            'asr_dec_dim': 256,              # ASR Decoder hidden size
            'asr_tied_dec_embs': False,      # ASR Share decoder embeddings
            'asr_att_temp': 1.0,             # ASR attention temperature
            'scheduling': 'random',          # Task scheduling type (random|alternating)
            'scheduling_p': (0.5, 0.5),      # Scheduling probabilities for 'random' mode
            'dropout': 0,                    # Generic dropout overall the architecture
            'max_len': None,        # Reject samples if len('bucket_by') > max_len
            'bucket_by': None,      # A key like 'en' to define w.r.t which dataset
                                    # the batches will be sorted
            'bucket_order': None,   # Can be 'ascending' or 'descending' to train
                                    # with increasing/decreasing sizes of sequences
            'direction': None,      # Network directionality, i.e. en->de
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

        # Only one source modality: Speech
        self.speech = str(self.topology.get_srcs('Kaldi')[0])

        targets = self.topology.get_trgs('Text')
        self.tl_asr, self.tl_slt = targets

        # Get vocabularies
        self.tl_asr_vocab = self.vocabs[self.tl_asr]
        self.tl_slt_vocab = self.vocabs[self.tl_slt]

        # Need to be set for early-stop evaluation
        self.val_refs = self.opts.data['val_set'][self.tl_slt]

        # Pick the correct encoder sampler method
        self.get_training_decoder = getattr(
            self, '_get_{}_decoder'.format(self.opts.model['scheduling']))
        self.scheduling_p = self.opts.model['scheduling_p']
        self.task_order = []

        self.ctx_dict = {self.speech: self.opts.model['sp_enc_dim'] * 2}

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
        for name, param in self.named_parameters():
            if param.requires_grad and 'bias' not in name:
                nn.init.kaiming_normal(param.data)

    def setup(self, is_train=True):
        self.enc = BiLSTMp(
            input_size=self.opts.model['sp_feat_dim'],
            hidden_size=self.opts.model['sp_enc_dim'],
            proj_size=self.opts.model['sp_proj_dim'],
            proj_activ=self.opts.model['sp_proj_activ'],
            dropout=self.opts.model['dropout'],
            layers=self.opts.model['sp_enc_layers'])

        asr_dec = ConditionalDecoder(
            input_size=self.opts.model['asr_emb_dim'],
            hidden_size=self.opts.model['asr_dec_dim'],
            n_vocab=len(self.tl_asr_vocab),
            rnn_type='gru',
            dec_init='zero',
            ctx_size_dict=self.ctx_dict,
            ctx_name=str(self.speech),
            att_temp=self.opts.model['asr_att_temp'],
            tied_emb=self.opts.model['asr_tied_dec_embs'],
            dropout_out=self.opts.model['dropout'])

        slt_dec = ConditionalDecoder(
            input_size=self.opts.model['slt_emb_dim'],
            hidden_size=self.opts.model['slt_dec_dim'],
            n_vocab=len(self.tl_slt_vocab),
            rnn_type='gru',
            dec_init='zero',
            ctx_size_dict=self.ctx_dict,
            ctx_name=str(self.speech),
            att_temp=self.opts.model['slt_att_temp'],
            tied_emb=self.opts.model['slt_tied_dec_embs'],
            dropout_out=self.opts.model['dropout'])

        self.decoders = ModuleDict({
            str(self.tl_asr): asr_dec,
            str(self.tl_slt): slt_dec,
        })

        self.decoder_names = list(self.decoders.keys())
        self.n_decoders = len(self.decoder_names)

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        dataset = MultimodalDataset(
            data=self.opts.data['{}_set'.format(split)],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model['max_len'],
            bucket_order=self.opts.model['bucket_order'])
        logger.info(dataset)
        return dataset

    def get_bos(self, batch_size):
        """Returns a representation for <bos> embeddings for decoding."""
        return torch.LongTensor(batch_size).fill_(1)

    def encode(self, batch, **kwargs):
        # Get the requested encoder
        return {str(self.speech): self.enc(batch[self.speech])}

    def _get_random_decoder(self, **kwargs):
        """Returns a random decoder with uniform probability by default."""
        if not any(self.task_order):
            # Buffer 10K samples for random scheduling
            order = np.random.choice(
                len(self.decoders), 10000, True, p=self.scheduling_p)
            self.task_order = [self.decoder_names[i] for i in order]

        # Return an decoder name
        return self.task_order.pop()

    def _get_alternating_decoder(self, **kwargs):
        """Returns the next decoder candidate based on update count."""
        return self.decoder_names[kwargs['uctr'] % self.n_decoders]

    def forward(self, batch, **kwargs):
        dec_id = kwargs.get('dec_id', None)
        if dec_id is None and self.training:
            dec_id = self.get_training_decoder(**kwargs)

        # Encode, decode, get loss and normalization factor
        enc = self.encode(batch)
        result = self.get_decoder(dec_id)(enc, batch[dec_id])
        result['n_items'] = torch.nonzero(batch[dec_id][1:]).shape[0]
        return result

    def test_performance(self, data_loader, dump_file=None):
        """Computes test set loss over the given DataLoader instance."""
        loss_dict = {k: Loss() for k in self.decoder_names}
        metrics = []

        for batch in data_loader:
            batch.to_gpu(volatile=True)
            for dec_id, loss in loss_dict.items():
                out = self.forward(batch, dec_id=dec_id)
                loss.update(out['loss'], out['n_items'])

        for dec_id, loss in loss_dict.items():
            loss = loss.get()
            logger.info('Task {} loss: {:.3f}'.format(dec_id, loss))
            metrics.append(loss)

        return [Metric('LOSS', np.mean(metrics), higher_better=False)]

    def get_decoder(self, task_id=None):
        return self.decoders[task_id]
