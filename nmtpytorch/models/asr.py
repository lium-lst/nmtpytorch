# -*- coding: utf-8 -*-
import logging

import torch
from torch import nn

from ..layers import BiLSTMp, ConditionalDecoder, FF
from ..datasets import MultimodalDataset
from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from . import NMT

logger = logging.getLogger('nmtpytorch')


# ASR with ESPNet style BiLSTMp encoder


class ASR(NMT):
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            'feat_dim': 43,                 # Speech features dimensionality
            'feat_transform': None,         # A FF to speech features: None, linear, tanh..
            'emb_dim': 300,                 # Decoder embedding dim
            'enc_dim': 320,                 # Encoder hidden size
            'enc_layers': '1_1_2_2_1_1',    # layer configuration
            'dec_dim': 320,                 # Decoder hidden size
            'proj_dim': 300,                # Intra-LSTM projection layer
            'proj_activ': 'tanh',           # Intra-LSTM projection activation
            'dec_type': 'gru',              # Decoder type (gru|lstm)
            'dec_init': 'mean_ctx',         # How to initialize decoder
                                            # (zero/mean_ctx/feats)
            'dec_init_size': None,          # feature vector dimensionality for
                                            # dec_init == 'feats'
            'dec_init_activ': 'tanh',       # Decoder initialization activation func
            'att_type': 'mlp',              # Attention type (mlp|dot)
            'att_temp': 1.,                 # Attention temperature
            'att_activ': 'tanh',            # Attention non-linearity (all torch nonlins)
            'att_mlp_bias': False,          # Enables bias in attention mechanism
            'att_bottleneck': 'hid',        # Bottleneck dimensionality (ctx|hid)
            'att_transform_ctx': True,      # Transform annotations before attention
            'dropout': 0,                   # Generic dropout overall the architecture
            'tied_dec_embs': False,         # Share decoder embeddings
            'max_len': None,                # Reject samples if len('bucket_by') > max_len
            'bucket_by': None,              # A key like 'en' to define w.r.t
                                            # which dataset batches will be sorted
            'bucket_order': None,           # Can be 'ascending' or 'descending'
                                            # for curriculum learning
                                            # NOTE: Noisy LSTM because of unhandled paddings
            'sampler_type': 'bucket',       # bucket or approximate
            'sched_sampling': 0,            # Scheduled sampling ratio
            'bos_type': 'emb',          #
            'bos_activ': None,          #
            'bos_dim': None,            #
            'direction': None,              # Network directionality, i.e. en->de
            'lstm_forget_bias': False,      # Initialize forget gate bias to 1 for LSTM
            'lstm_bias_zero': False,        # Use zero biases for LSTM
            'adaptation': False,            # Enable/disable AM adaptation
            'adaptation_type': 'early',     # Kept for backward-compatibility
            'adaptation_dim': None,         # Input dim for auxiliary feat vectors
            'adaptation_activ': None,       # Non-linearity for adaptation FF
            'io_bias': 0.1,                 # bias for IO adaptation
        }

    def __init__(self, opts):
        # Don't call NMT init as it's too different from ASR
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

        # Load vocabularies here
        for name, fname in self.opts.vocabulary.items():
            self.vocabs[name] = Vocabulary(fname, name=name)

        # Inherently non multi-lingual aware
        self.src = self.topology.first_src

        self.tl = self.topology.first_trg
        self.trg_vocab = self.vocabs[self.tl]
        self.n_trg_vocab = len(self.trg_vocab)

        # Context size is enc_dim because of proj layers
        self.ctx_sizes = {str(self.src): self.opts.model['enc_dim']}

        # Need to be set for early-stop evaluation
        # NOTE: This should come from config or elsewhere
        self.val_refs = self.opts.data['val_set'][self.tl]

    def reset_parameters(self):
        # Use kaiming normal for everything as it is a sane default
        # Do not touch biases for now
        for name, param in self.named_parameters():
            if param.requires_grad and 'bias' not in name:
                nn.init.kaiming_normal_(param.data)

        if self.opts.model['lstm_bias_zero'] or \
                self.opts.model['lstm_forget_bias']:
            for name, param in self.speech_enc.named_parameters():
                if 'bias_hh' in name or 'bias_ih' in name:
                    # Reset bias to 0
                    param.data.fill_(0.0)
                    if self.opts.model['lstm_forget_bias']:
                        # Reset forget gate bias of LSTMs to 1
                        # the tensor organized as: inp,forg,cell,out
                        n = param.numel()
                        param[n // 4: n // 2].data.fill_(1.0)

    def setup(self, is_train=True):
        self.speech_enc = BiLSTMp(
            input_size=self.opts.model['feat_dim'],
            hidden_size=self.opts.model['enc_dim'],
            proj_size=self.opts.model['proj_dim'],
            proj_activ=self.opts.model['proj_activ'],
            dropout=self.opts.model['dropout'],
            layers=self.opts.model['enc_layers'])

        ################
        # Create Decoder
        ################
        self.dec = ConditionalDecoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name=str(self.src),
            tied_emb=self.opts.model['tied_dec_embs'],
            dec_init=self.opts.model['dec_init'],
            dec_init_size=self.opts.model['dec_init_size'],
            dec_init_activ=self.opts.model['dec_init_activ'],
            att_type=self.opts.model['att_type'],
            att_temp=self.opts.model['att_temp'],
            att_activ=self.opts.model['att_activ'],
            transform_ctx=self.opts.model['att_transform_ctx'],
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout'],
            sched_sample=self.opts.model['sched_sampling'],
            bos_type=self.opts.model['bos_type'],
            bos_dim=self.opts.model['bos_dim'],
            bos_activ=self.opts.model['bos_activ'])

        if self.opts.model['adaptation']:
            out_dim = self.opts.model['feat_dim']
            if self.opts.model['adaptation_type'].startswith('early'):
                # Simple single layer
                self.vis_proj = FF(self.opts.model['adaptation_dim'],
                                   out_dim,
                                   activ=self.opts.model['adaptation_activ'],
                                   bias=False)
            elif self.opts.model['adaptation_type'] == 'deep':
                # 3 layers of 512d with sigmoid NL and one output layer
                activ = self.opts.model['adaptation_activ']
                self.vis_proj = nn.Sequential(
                    FF(self.opts.model['adaptation_dim'], 256, activ=activ),
                    FF(256, 256, activ=activ),
                    FF(256, 256, activ=activ),
                    FF(256, out_dim, activ=None),
                )
            elif self.opts.model['adaptation_type'] == 'io':
                self.emb_cat = nn.Embedding(3, out_dim, padding_idx=2)

        if self.opts.model['feat_transform']:
            self.feat_transform = FF(self.opts.model['feat_dim'],
                                     self.opts.model['feat_dim'], bias=False,
                                     activ=self.opts.model['feat_transform'])

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
        # Speech features -> x
        x = batch[self.src]
        if self.opts.model['feat_transform']:
            x = self.feat_transform(x)

        if self.opts.model['adaptation']:
            if self.opts.model['adaptation_type'] == 'io':
                x *= (torch.sigmoid(self.emb_cat(batch['io'])) + self.opts.model['io_bias'])
            elif self.opts.model['adaptation_type'] == 'early_mul':
                x *= (torch.sigmoid(self.vis_proj(batch['feats'])) + self.opts.model['io_bias'])
            else:
                x += self.vis_proj(batch['feats'])

        d = {str(self.src): self.speech_enc(x)}
        if 'feats' in batch:
            d['feats'] = (batch['feats'], None)
        return d
