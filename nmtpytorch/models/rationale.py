# -*- coding: utf-8 -*-
from ..logger import Logger

import torch
from torch import nn
from torch import distributions

from ..layers import FF, ArgSelect, Pool
from ..layers.decoders import get_decoder
from ..datasets import MultimodalDataset

from . import NMT

log = Logger()


class Rationale(NMT):
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            ### Generator
            'gen_dim': 256,             # Generator hidden size
            'gen_type': 'gru',          # Generator type (gru|lstm)
            'gen_bidir': True,          # Bi-directional encoder in generator
            'gen_n_layers': 1,          # Number of stacked encoders in the generator
            'gen_sel_type': 'ind',      # Independent selection
            ### Encoder
            'enc_dim': 256,             # Re-encoder hidden size
            'enc_type': 'gru',          # Encoder type (gru|lstm|avg|sum|max)
            'enc_bidir': True,          # Bi-directional encoder in re-encoder
            'enc_method': 'proper',     # If enc_type is recurrent:
                                        #  proper: re-encode short sequence
                                        #  masked: re-encode 0-masked sequence
            'enc_n_layers': 1,          # If enc_type is recurrent: # of layers
            ### Decoder
            'dec_dim': 256,             # Decoder hidden size
            'dec_type': 'gru',          # Decoder type (gru|lstm)
            'dec_variant': 'vector',    # (cond|simplegru|vector)
            'dec_init': 'mean_ctx',     # How to initialize decoder (zero/mean_ctx/feats)
            ### Generic S2S arguments
            'direction': None,          # Network directionality, i.e. en->de
            'max_len': 80,              # Reject sentences where 'bucket_by' length > 80
            'bucket_by': None,          # A key like 'en' to define w.r.t which dataset
                                        # the batches will be sorted
            'bucket_order': None,       # Curriculum: ascending/descending/None
            'sampler_type': 'bucket',   # bucket or approximate
            ### Other arguments
            'proj_embs': False,         # Additional embedding projection
            'pretrained_embs': '',      # optional path to embeddings tensor
            'freeze_embs': True,        # Freeze embeddings
            'short_list': 100000000,    # Use all pretrained vocab
            'dropout': 0,               # Simple dropout layer
            'add_src_eos': True,        # Append <eos> or not to inputs
            'rnn_dropout': 0,           # RNN dropout if *_n_layers > 1
            'emb_dim': 128,             # Source and target embedding sizes
            'enc_gen_share': False,     # Share RNNs of generator and re-encoder
            'tied_emb': False,          # Share embeddings: (False|2way|3way)
            'lambda_coherence': 0,      # Coherence penalty
            'lambda_sparsity': 0,       # Sparsity penalty
            'n_samples': 1,             # How many samples to get from generator
        }

    def __init__(self, opts):
        super().__init__(opts)

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        self.dataset = MultimodalDataset(
            data=self.opts.data['{}_set'.format(split)],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model['max_len'],
            bucket_order=self.opts.model['bucket_order'],
            sampler_type=self.opts.model['sampler_type'],
            eos=self.opts.model['add_src_eos'])
        log.log(self.dataset)
        return self.dataset

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if name == 'embs.weight' and self.opts.model['pretrained_embs']:
                continue
            # Skip 1-d biases and scalars
            if param.requires_grad and param.dim() > 1:
                nn.init.kaiming_normal_(param.data)

        # Reset padding embedding to 0
        with torch.no_grad():
            self.embs.weight.data[0].fill_(0)

    def setup(self, is_train=True):
        """Sets up NN topology by creating the layers."""
        # Source embeddings
        if self.opts.model['pretrained_embs']:
            embs = torch.load(self.opts.model['pretrained_embs']).float()
            embs = embs[:self.opts.model['short_list']]
            self.embs = nn.Embedding.from_pretrained(
                embs, freeze=self.opts.model['freeze_embs'], padding_idx=0)
            self.opts.model['emb_dim'] = embs.shape[1]
            if self.opts.model['freeze_embs']:
                self.opts.train['freeze_layers'] = 'embs.'
        else:
            self.embs = nn.Embedding(
                self.n_src_vocab, self.opts.model['emb_dim'], padding_idx=0)

        if self.opts.model['proj_embs']:
            self.proj_embs = FF(
                self.opts.model['emb_dim'], self.opts.model['emb_dim'])

        # Generic dropout layer
        self.do = nn.Dropout(self.opts.model['dropout'])

        ##################
        # Generator block
        ##################
        layers = []
        RNN = getattr(nn, self.opts.model['gen_type'].upper())

        # RNN Encoder
        layers.append(RNN(
            self.opts.model['emb_dim'], self.opts.model['gen_dim'],
            self.opts.model['gen_n_layers'], batch_first=False,
            dropout=self.opts.model['rnn_dropout'],
            bidirectional=self.opts.model['gen_bidir']))
        # Return the sequence of hidden outputs
        layers.append(ArgSelect(0))

        # Create the generator wrapper
        self.gen = nn.Sequential(*layers)

        self.n_samples = torch.ones((self.opts.model['n_samples'], 1, 1))

        # Independent selection
        gen_hid_dim = self.opts.model['gen_dim'] * (int(self.opts.model['gen_bidir']) + 1)
        if self.opts.model['gen_sel_type'] == 'ind':
            self.z_layer = FF(gen_hid_dim, 1, activ='sigmoid')
        # Dependent selection
        elif self.opts.model['gen_sel_type'] == 'dep':
            self.z_layer = None
            raise NotImplementedError('Dependent selection not implemented')

        ###################
        # Create Re-encoder
        ###################
        if self.opts.model['enc_gen_share']:
            self.enc = self.gen
        else:
            layers = []
            enc_type = self.opts.model['enc_type'].upper()
            is_rnn = enc_type in ('GRU', 'LSTM')
            self.ctx_size = self.opts.model['enc_dim']
            if is_rnn:
                self.ctx_size *= int(self.opts.model['enc_bidir']) + 1

                RNN = getattr(nn, enc_type)

                # RNN Re-encoder
                layers.append(RNN(
                    self.opts.model['emb_dim'], self.opts.model['gen_dim'],
                    self.opts.model['gen_n_layers'], batch_first=False,
                    dropout=self.opts.model['rnn_dropout'],
                    bidirectional=self.opts.model['gen_bidir']))
                # Return the sequence of hidden outputs
                layers.append(ArgSelect(0))
            else:
                # Pool generator's hidden states
                layers.append(
                    Pool(self.opts.model['enc_type'], pool_dim=0))

        # Create the re-encoder wrapper
        self.enc = nn.Sequential(*layers)

        ################
        # Create Decoder
        ################
        Decoder = get_decoder(self.opts.model['dec_variant'])
        self.dec = Decoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict={str(self.sl): self.ctx_size},
            ctx_name=str(self.sl),
            dec_init='mean_ctx',
#             dec_init_activ='tanh',
            # dec_init_size=self.ctx_size,
            tied_emb=self.opts.model['tied_emb'],
            dropout_out=self.opts.model['dropout'])

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            self.embs.weight = self.dec.emb.weight

    def encode(self, batch, **kwargs):
        # Fetch embeddings -> T x B x D
        embs = self.embs(batch[str(self.sl)])

        # Dropout and pass embeddings through RNN -> T x B x G
        hs = self.gen(self.do(embs))

        # Apply a sigmoid layer and transpose -> B x T
        p_z = self.z_layer(hs).squeeze(-1).t()

        # Create a distribution
        dist = distributions.Binomial(total_count=self.n_samples, probs=p_z)

        # Draw N samples from it -> N x B x T
        z = dist.sample()

        # Mask out non-rationale bits and pool -> 1, B, G
        sent_rep = self.enc(z.permute((2, 1, 0)) * embs)
        return {str(self.sl): (sent_rep, None)}

    def forward(self, batch, **kwargs):
        result = self.dec(self.encode(batch), batch[self.tl])
        result['n_items'] = torch.nonzero(batch[self.tl][1:]).shape[0]
        return result
