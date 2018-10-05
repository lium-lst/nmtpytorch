# -*- coding: utf-8 -*-
import logging

from torch import nn

from ..layers import ReverseVideoDecoder, FeatureEncoder
from ..utils.misc import get_n_params
from ..utils.topology import Topology
from ..utils.ml_metrics import Loss
from ..datasets import MultimodalDataset
from ..metrics import Metric

logger = logging.getLogger('nmtpytorch')


class VideoReconstruction(nn.Module):
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            'emb_dim': 256,             # Source and target embedding sizes
            'vid_dim': 2048,            # Video frame input size
            'proj_size': 512,           # Video frame embedding size
            'emb_maxnorm': None,        # Normalize embeddings l2 norm to 1
            'emb_gradscale': False,     # Scale embedding gradients w.r.t. batch frequency
            'enc_dim': 512,             # Encoder hidden size
            'enc_type': 'gru',          # Encoder type (gru|lstm)
            'n_encoders': 1,            # Number of stacked encoders
            'dec_dim': 512,             # Decoder hidden size
            'dec_type': 'gru',          # Decoder type (gru|lstm)
            'dec_init': 'mean_ctx',     # How to initialize decoder (zero/mean_ctx/feats)
            'dec_init_size': None,      # feature vector dimensionality for
                                        # dec_init == 'feats'
            'att_type': 'mlp',          # Attention type (mlp|dot)
            'att_temp': 1.,             # Attention temperature
            'att_activ': 'tanh',        # Attention non-linearity (all torch nonlins)
            'att_mlp_bias': False,      # Enables bias in attention mechanism
            'att_bottleneck': 'ctx',    # Bottleneck dimensionality (ctx|hid)
            'att_transform_ctx': True,  # Transform annotations before attention
            'bidirectional': True,      # Whether the encoder is bidirectional or not
            'dropout_emb': 0,           # Simple dropout to source embeddings
            'dropout_ctx': 0,           # Simple dropout to source encodings
            'dropout_out': 0,           # Simple dropout to decoder output
            'dropout_enc': 0,           # Intra-encoder dropout if n_encoders > 1
            'tied_emb': False,          # Share embeddings: (False|2way|3way)
            'direction': None,          # Network directionality, i.e. en->de
            'max_len': 80,              # Reject sentences where 'bucket_by' length > 80
            'bucket_by': None,          # A key like 'en' to define w.r.t which dataset
                                        # the batches will be sorted
            'bucket_order': None,       #
            'loss_type': 'MSE_loss',    # Loss type (MSE_loss | SmoothL1)
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

        # Textual context size is equal enc_dim * 2 using bidirectional RNN
        # since it is the concatenation of forward and backward hidden states
        if 'enc_dim' in self.opts.model:
            if self.opts.model['bidirectional']:
                self.ctx_sizes = {'feats_in': self.opts.model['enc_dim'] * 2}
            else:
                self.ctx_sizes = {'feats_in': self.opts.model['enc_dim']}

        if 'SmoothL1' in self.opts.model['loss_type']:
            self.use_smoothL1 = True

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
        """Sets up NN topology by creating the layers."""
        ########################
        # Create Video  Encoder
        ########################
        self.enc = FeatureEncoder(
            input_size=self.opts.model['vid_dim'],
            proj_size=self.opts.model['proj_size'],
            hidden_size=self.opts.model['enc_dim'],
            rnn_type=self.opts.model['enc_type'],
            bidirectional=self.opts.model['bidirectional'],
            dropout_emb=self.opts.model['dropout_emb'],
            dropout_ctx=self.opts.model['dropout_ctx'],
            dropout_rnn=self.opts.model['dropout_enc'],
            num_layers=self.opts.model['n_encoders'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'])

        ################
        # Create Decoder
        ################
        self.dec = ReverseVideoDecoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            rnn_type=self.opts.model['dec_type'],
            video_dim=self.opts.model['vid_dim'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name='feats_in',
            dec_init=self.opts.model['dec_init'],
            dec_init_size=self.opts.model['dec_init_size'],
            att_type=self.opts.model['att_type'],
            att_temp=self.opts.model['att_temp'],
            att_activ=self.opts.model['att_activ'],
            transform_ctx=self.opts.model['att_transform_ctx'],
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout_out'],
            use_smoothL1=self.opts.model['loss_type'])

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        dataset = MultimodalDataset(
            data=self.opts.data['{}_set'.format(split)], vocabs={},
            mode=mode, topology=self.topology, batch_size=batch_size,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model['max_len'],
            bucket_order=self.opts.model['bucket_order'])
        logger.info(dataset)
        return dataset

    def encode(self, batch, **kwargs):
        """Encodes all inputs and returns a dictionary.

        Arguments:
            batch (dict): A batch of samples with keys designating the
                information sources
        """

        return {'feats_in': self.enc(batch['feats_in'])}

    def forward(self, batch, **kwargs):
        """Computes the forward-pass of the network and returns batch loss.

        Arguments:
            batch (dict): A batch of samples with keys designating the source
                and target modalities.

        """
        result = self.dec(self.encode(batch), batch, use_smoothL1=self.training)

        return result

    def test_performance(self, data_loader, dump_file=None):
        """Computes test set loss over the given DataLoader instance."""
        loss = Loss()

        for batch in data_loader:
            batch.to_gpu(volatile=True)
            out = self.forward(batch, use_SmoothL1=False)
            loss.update(out['loss'], out['n_items'])

        return [
            Metric('LOSS', loss.get(), higher_better=False)
        ]
