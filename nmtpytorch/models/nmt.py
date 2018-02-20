# -*- coding: utf-8 -*-
import torch.nn as nn

from ..vocabulary import Vocabulary
from ..layers import TextEncoder
from ..layers import ConditionalDecoder
from ..utils.data import to_var
from ..utils.topology import Topology

from ..datasets import BitextDataset


class NMT(nn.Module):
    r"""A sequence-to-sequence NMT model.

    Encoder RNN's will use CuDNN's optimized versions with packing and
    padding. The decoder currently used is ConditionalDecoder which is a
    Conditional GRU architecture unrolled manually.
    """
    def __init__(self, opts, logger=None):
        """This should only be used for argument processing."""
        super().__init__()

        self.aux_loss = None

        self.vocabs = {}
        self.datasets = {}
        self.logger = logger
        self.print = print if self.logger is None else self.logger.info

        # Get a copy of model options coming from config
        # Consume them with .pop(), integrating model defaults
        # What is left inside kwargs -> Unused arguments
        kwargs = opts.model.copy()

        # Get direction and parse it
        opts.model['direction'] = kwargs.pop('direction')
        self.topology = Topology(opts.model['direction'])

        # Load vocabularies here
        for name, fname in opts.vocabulary.items():
            self.vocabs[name] = Vocabulary(fname, name=name)

        self.sl = self.topology.get_src_langs()[0]
        self.tl = self.topology.get_trg_langs()[0]

        self.src_vocab = self.vocabs[self.sl]
        self.trg_vocab = self.vocabs[self.tl]

        self.n_src_vocab = len(self.src_vocab)
        self.n_trg_vocab = len(self.trg_vocab)

        opts.model['emb_dim'] = kwargs.pop('emb_dim', 256)
        opts.model['enc_dim'] = kwargs.pop('enc_dim', 512)
        opts.model['dec_dim'] = kwargs.pop('dec_dim', 1024)
        opts.model['enc_type'] = kwargs.pop('enc_type', 'gru')
        opts.model['dec_type'] = kwargs.pop('dec_type', 'gru')
        opts.model['n_encoders'] = kwargs.pop('n_encoders', 1)

        # How to initialize decoder (zero/mean_ctx)
        opts.model['dec_init'] = kwargs.pop('dec_init', 'mean_ctx')
        opts.model['att_type'] = kwargs.pop('att_type', 'mlp')

        # Various dropouts
        opts.model['dropout_emb'] = kwargs.pop('dropout_emb', 0.)
        opts.model['dropout_ctx'] = kwargs.pop('dropout_ctx', 0.)
        opts.model['dropout_out'] = kwargs.pop('dropout_out', 0.)
        opts.model['dropout_enc'] = kwargs.pop('dropout_enc', 0.)

        # Tie embeddings: False/2way/3way
        opts.model['tied_emb'] = kwargs.pop('tied_emb', False)

        # Filter out sentence pairs where target >= 80 tokens
        opts.model['max_trg_len'] = kwargs.pop('max_trg_len', 80)

        # Sanity check after consuming all arguments
        if len(kwargs) > 0:
            self.print('Unused model args: {}'.format(','.join(kwargs.keys())))

        # Check vocabulary sizes for 3way tying
        if opts.model['tied_emb'] == '3way':
            assert self.n_src_vocab == self.n.trg_vocab, \
                "The vocabulary sizes do not match for 3way tied embeddings."

        # Context size is always equal to encoder hidden dim * 2 since
        # it is the concatenation of forward and backward hidden states
        self.ctx_size = opts.model['enc_dim'] * 2

        # Set some shortcuts
        self.opts = opts

        # Need to be set for early-stop evaluation
        self.val_refs = self.opts.data['val_set'][self.tl]

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad and 'bias' not in name:
                nn.init.kaiming_normal(param.data)

    def setup(self, reset_params=True):
        """Sets up NN topology by creating the layers."""
        ################
        # Create encoder
        ################
        self.enc = TextEncoder(input_size=self.opts.model['emb_dim'],
                               hidden_size=self.opts.model['enc_dim'],
                               n_vocab=self.n_src_vocab,
                               cell_type=self.opts.model['enc_type'],
                               dropout_emb=self.opts.model['dropout_emb'],
                               dropout_ctx=self.opts.model['dropout_ctx'],
                               dropout_rnn=self.opts.model['dropout_enc'],
                               num_layers=self.opts.model['n_encoders'])

        ################
        # Create Decoder
        ################
        self.dec = ConditionalDecoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            ctx_size=self.ctx_size,
            n_vocab=self.n_trg_vocab,
            tied_emb=self.opts.model['tied_emb'],
            dec_init=self.opts.model['dec_init'],
            att_type=self.opts.model['att_type'],
            dropout_out=self.opts.model['dropout_out'])

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            self.enc.emb.weight = self.dec.emb.weight

        if reset_params:
            self.reset_parameters()

    def load_data(self, split):
        """Loads the requested dataset split."""
        if split not in self.datasets:
            # Only for training
            max_trg_len = None
            if split == 'train':
                max_trg_len = self.opts.model['max_trg_len']
            dataset = BitextDataset(split=split,
                                    data_dict=self.opts.data,
                                    vocabs=self.vocabs,
                                    topology=self.topology,
                                    max_trg_len=max_trg_len)
            self.datasets[split] = dataset
            self.print(dataset)

    def compute_loss(self, data_loader):
        """Computes test set loss over the given DataLoader instance."""
        total_loss = 0.0
        n_tokens_seen = 0

        for batch in data_loader:
            loss = self.forward(to_var(batch, volatile=True))
            total_loss += (loss * self.n_tokens)
            n_tokens_seen += self.n_tokens

        return total_loss.data.cpu()[0] / n_tokens_seen

    def encode(self, data):
        """Encodes all inputs and returns a dictionary.

        Arguments:
            data (dict): A batch of samples with keys designating the source
                and target modalities.

        Returns:
            dict:
                A dictionary where keys are source modalities compatible
                with the data loader and the values are tuples where the
                elements are encodings and masks. The mask can be ``None``
                if the relevant modality does not require a mask.
        """
        return {'txt': self.enc(data[self.sl])}

    def forward(self, data):
        """Computes the forward-pass of the network and returns batch loss.

        Arguments:
            data (dict): A batch of samples with keys designating the source
                and target modalities.

        Returns:
            Variable:
                A scalar loss normalized w.r.t batch size and token counts.

        Note:
            This method should assign the number of target tokens for which
            the loss is computed inside ``self.n_tokens`` in order for the
            ``MainLoop`` to correctly compute epoch loss.
        """

        # Get all source encodings
        ctx_dict = self.encode(data)

        # Save number of target tokens for loss normalization
        self.n_tokens = data[self.tl][1:].numel()

        # Get final loss as a sum
        return self.dec(ctx_dict, data[self.tl]) / self.n_tokens
