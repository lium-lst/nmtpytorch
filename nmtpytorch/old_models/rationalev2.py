# -*- coding: utf-8 -*-
from ..logger import Logger

import torch
from torch import nn
from torch import distributions
from torch.nn import functional as F

from ..layers import FF, ArgSelect, Pool, PEmbedding
from ..layers.decoders import ConditionalDecoder
from ..utils.nn import get_rnn_hidden_state
from ..datasets import MultimodalDataset

from . import NMT

import ipdb

log = Logger()


class VectorDecoder(ConditionalDecoder):
    """Single-layer RNN decoder using fixed-size vector representation."""
    def __init__(self, **kwargs):
        # Disable attention
        kwargs['att_type'] = None
        super().__init__(**kwargs)

    def f_next(self, ctx_dict, y, h):
        """Applies one timestep of recurrence."""
        # Get hidden states from the decoder
        h1_c1 = self.dec0(y, self._rnn_unpack_states(h))
        h1 = get_rnn_hidden_state(h1_c1)

        # Project hidden state to embedding size
        o = self.hid2out(h1)

        # Apply dropout if any
        logit = self.do_out(o) if self.dropout_out > 0 else o

        # Transform logit to T*B*V (V: vocab_size)
        # Compute log_softmax over token dim
        logit = logit @ self.out2prob.proj(self.out2prob.weight).t()
        log_p = F.log_softmax(logit, dim=-1)

        # Return log probs and new hidden states
        return log_p, self._rnn_pack_states(h1_c1)


class Rationalev2(NMT):
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
            'emb_dim': 128,             # Source and target embedding sizes
            'proj_emb_dim': 128,        # Additional embedding projection output
            'proj_emb_activ': 'linear', # Additional embedding projection non-lin
            'pretrained_embs': '',      # optional path to embeddings tensor
            'pretrained_embs_l2': False,# L2 normalize pretrained embs
            'short_list': 100000000,    # Use all pretrained vocab
            'dropout': 0,               # Simple dropout layer
            'add_src_eos': True,        # Append <eos> or not to inputs
            'rnn_dropout': 0,           # RNN dropout if *_n_layers > 1
            'lambda_coherence': 0,      # Coherence penalty
            'lambda_sparsity': 0,       # Sparsity penalty
            'n_samples': 1,             # How many samples to get from generator
            'tied_emb': False,          # Not used, always tied in this model
        }

    def __init__(self, opts):
        super().__init__(opts)

        # Clip to short_list if given
        self.n_src_vocab = min(self.n_src_vocab, self.opts.model['short_list'])
        self.n_trg_vocab = min(self.n_trg_vocab, self.opts.model['short_list'])

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
            # Skip 1-d biases and scalars
            if param.requires_grad and param.dim() > 1:
                nn.init.kaiming_normal_(param.data)

        with torch.no_grad():
            if self.opts.model['pretrained_embs']:
                embs = torch.load(self.opts.model['pretrained_embs'])
                embs = embs[:self.opts.model['short_list']].float()
                if self.opts.model['pretrained_embs_l2']:
                    embs.div_(embs.norm(p=2, dim=-1, keepdim=True))
                self.embs.weight.data.copy_(embs)

            # Reset padding embedding to 0
            self.embs.weight.data[0].fill_(0)

        # Use the same PEmbedding for the decoder as well
        self.dec.emb = self.embs
        self.dec.out2prob = self.embs

    def setup(self, is_train=True):
        """Sets up NN topology by creating the layers."""
        # Create embeddings followed by projection layer
        self.embs = PEmbedding(
            self.n_src_vocab, self.opts.model['emb_dim'],
            out_dim=self.opts.model['proj_emb_dim'],
            activ=self.opts.model['proj_emb_activ'])

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

            # Create the re-encoder wrapper
            self.enc = nn.Sequential(*layers)
        else:
            # Poolings will be done in the decoder
            self.enc = lambda x: x

        ################
        # Create Decoder
        ################
        self.dec = VectorDecoder(
            input_size=self.opts.model['proj_emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict={str(self.sl): self.ctx_size},
            ctx_name=str(self.sl),
            dec_init=self.opts.model['dec_init'],
            dropout_out=self.opts.model['dropout'])

        if not is_train:
            # NOTE: nmtpy translate does not call reset_parameters()
            # Use the same PEmbedding for the decoder as well when decoding
            self.dec.emb = self.embs
            self.dec.out2prob = self.embs

    def encode(self, batch, **kwargs):
        # Fetch embeddings -> T x B x D
        embs = self.embs(batch[str(self.sl)])

        # Dropout and pass embeddings through RNN -> T x B x G
        hs = self.gen(self.do(embs))

        # Apply a sigmoid layer and transpose -> B x T
        p_z = self.z_layer(hs).squeeze(-1).t()

        # Create a distribution
        dist = distributions.Binomial(total_count=self.n_samples, probs=p_z)

        # Draw N samples from it -> N x B x T -> T x B x N
        z = dist.sample().permute((2, 1, 0))

        # Mask out non-rationale bits and re-encode -> 1, B, G
        sent_rep = self.enc(z * embs)

        return {str(self.sl): (sent_rep, None)}

    def forward(self, batch, **kwargs):
        result = self.dec(self.encode(batch), batch[self.tl])
        result['n_items'] = torch.nonzero(batch[self.tl][1:]).shape[0]
        return result
