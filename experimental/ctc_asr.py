# -*- coding: utf-8 -*-
import logging

from torch import nn

from ..layers.speech import BiRNNPv1
from ..layers import CTCDecoder
from ..utils.misc import get_n_params
from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from ..utils.ml_metrics import Loss
from ..datasets import MultimodalDataset
from ..metrics import Metric

logger = logging.getLogger('nmtpytorch')


class CTCASR(nn.Module):
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            'feat_dim': 43,             # Speech features dimensionality
            'enc_dim': 256,             # Encoder hidden size
            'enc_type': 'gru',          # Encoder type (gru|lstm)
            'enc_subsample': (),        # Tuple of subsampling factors
                                        # Also defines # of subsampling layers
            'n_sub_layers': 1,          # Number of stacked RNNs in each subsampling block
            'n_base_encoders': 1,       # Number of stacked encoders
            'dropout': 0,               # Generic dropout overall the architecture
            'max_len': None,            # Reject samples if len('bucket_by') > max_len
            'bucket_by': None,          # A key like 'en' to define w.r.t which dataset
                                        # the batches will be sorted
            'bucket_order': None,       # Can be 'ascending' or 'descending' to train
                                        # with increasing/decreasing sizes of sequences
            'direction': None,          # Network directionality, i.e. en->de
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

        # Inherently non multi-lingual aware
        self.src = self.topology.first_src

        self.tl = self.topology.first_trg
        self.trg_vocab = self.vocabs[self.tl]
        self.n_trg_vocab = len(self.trg_vocab)

        # Context size is always equal to enc_dim * 2 since
        # it is the concatenation of forward and backward hidden states
        if 'enc_dim' in self.opts.model:
            self.ctx_sizes = {str(self.src): self.opts.model['enc_dim'] * 2}

        # Need to be set for early-stop evaluation
        # NOTE: This should come from config or elsewhere
        self.val_refs = self.opts.data['val_set'][self.tl]

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
        self.speech_enc = BiRNNPv1(
            input_size=self.opts.model['feat_dim'],
            hidden_size=self.opts.model['enc_dim'],
            rnn_type=self.opts.model['enc_type'],
            dropout=self.opts.model['dropout'],
            subsample=self.opts.model['enc_subsample'],
            num_sub_layers=self.opts.model['n_sub_layers'],
            num_base_layers=self.opts.model['n_base_encoders'])

        ################
        # Create Decoder
        ################
        self.dec = CTCDecoder(
            input_size=self.ctx_sizes[str(self.src)],
            n_vocab=self.n_trg_vocab,
            ctx_name=str(self.src))

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

    def encode(self, batch):
        return {str(self.src): self.speech_enc(batch[self.src])}

    def forward(self, batch, **kwargs):
        n_frames, batch_size, _ = batch[self.src].shape

        # Let's just skip <bos> position
        y = batch[self.tl][1:]

        # We shift all labels by 1 so that 0 becomes a reserved symbol for blank
        result = self.dec(self.encode(batch), y + 1)
        result['n_items'] = n_frames * batch_size
        return result

    def test_performance(self, data_loader, dump_file=None):
        """Computes test set loss over the given DataLoader instance."""
        loss = Loss()

        for batch in data_loader:
            batch.to_gpu(volatile=True)
            out = self.forward(batch)
            loss.update(out['loss'], out['n_items'])

        return [
            Metric('LOSS', loss.get(), higher_better=False),
        ]
