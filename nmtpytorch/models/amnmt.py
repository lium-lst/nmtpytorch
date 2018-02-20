# -*- coding: utf-8 -*-
import torch.nn as nn

from ..vocabulary import Vocabulary
from ..layers import ImageEncoder, TextEncoder, ConditionalMMDecoder
from ..utils.nn import set_learnable
from ..utils.topology import Topology
from ..utils.data import to_var

from ..datasets import Multi30kDataset


class AttentiveMNMT(nn.Module):
    r"""A sequence-to-sequence NMT model with visual attention."""
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

        # CNN stuff
        opts.model['img_mode'] = kwargs.pop('img_mode', 'raw')
        if opts.model['img_mode'] == 'raw':
            assert 'cnn_type' in opts.model, "cnn_type not provided"
            assert 'cnn_layer' in opts.model, "cnn_layer not provided"
            opts.model['cnn_trainable'] = kwargs.pop('cnn_trainable', False)
            opts.model['cnn_pretrained'] = kwargs.pop('cnn_pretrained', True)
            opts.model['cnn_type'] = kwargs.pop('cnn_type', 'resnet50')
            opts.model['cnn_layer'] = kwargs.pop('cnn_layer', 'res4f_relu')
        else:
            # Make this fail to make it mandatory
            assert 'img_ctx_dim' in opts.model, \
                "You should provide img_ctx_dim for the given conv features."

            opts.model['img_ctx_dim'] = kwargs.pop('img_ctx_dim')

        # How to fusion multimodal contexts (sum/mul/concat)
        opts.model['fusion_type'] = kwargs.pop('fusion_type', 'sum')

        opts.model['emb_dim'] = kwargs.pop('emb_dim', 256)
        opts.model['enc_dim'] = kwargs.pop('enc_dim', 512)
        opts.model['dec_dim'] = kwargs.pop('dec_dim', 1024)
        opts.model['enc_type'] = kwargs.pop('enc_type', 'gru')
        opts.model['dec_type'] = kwargs.pop('dec_type', 'gru')
        opts.model['n_encoders'] = kwargs.pop('n_encoders', 1)

        # How to initialize decoder (zero/mean_ctx)
        opts.model['dec_init'] = kwargs.pop('dec_init', 'mean_ctx')

        # Attention related parameters
        opts.model['att_type'] = kwargs.pop('att_type', 'mlp')

        # The bottleneck dimension during attention computation
        opts.model['att_bottleneck'] = kwargs.pop('att_bottleneck', 'ctx')
        opts.model['att_activ'] = kwargs.pop('att_activ', 'tanh')

        # Various dropouts
        opts.model['dropout_emb'] = kwargs.pop('dropout_emb', 0.)
        opts.model['dropout_ctx'] = kwargs.pop('dropout_ctx', 0.)
        opts.model['dropout_out'] = kwargs.pop('dropout_out', 0.)
        opts.model['dropout_enc'] = kwargs.pop('dropout_enc', 0.)

        # Tie embeddings: False/2way/3way
        opts.model['tied_emb'] = kwargs.pop('tied_emb', False)

        # Sanity check after consuming all arguments
        if len(kwargs) > 0:
            self.print('Unused model args: {}'.format(','.join(kwargs.keys())))

        # Check vocabulary sizes for 3way tying
        if opts.model['tied_emb'] == '3way':
            assert self.n_src_vocab == self.n.trg_vocab, \
                "The vocabulary sizes do not match for 3way tied embeddings."

        # Textual context size is always equal to enc_dim * 2 since
        # it is the concatenation of forward and backward hidden states
        self.txt_ctx_dim = opts.model['enc_dim'] * 2

        # Set some shortcuts
        self.opts = opts

        # Need to be set for early-stop evaluation
        self.val_refs = self.opts.data['val_set'][self.tl]

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad and 'bias' not in name:
                nn.init.kaiming_normal(param.data)

    def setup_cnn(self):
        def process_raw_image(image):
            # Extract convolutional features
            # Section 3.1: a = {a_1 ... a_L}, a_i \in \mathrm{R}^D
            img_ctx = self.cnn(image).view(
                (image.shape[0], self.opts.model['img_ctx_dim'], -1))
            # make them (batch_size, w*h, ctx_dim)
            img_ctx.transpose_(1, 2)
            return img_ctx

        # Get CNN encoder for the images
        if self.opts.model['img_mode'] == 'raw':
            self.print('Loading CNN')
            cnn_encoder = ImageEncoder(
                cnn_type=self.opts.model['cnn_type'],
                pretrained=self.opts.model['cnn_pretrained'])
            self.cnn, dims = cnn_encoder.get(self.opts.model['cnn_layer'])
            self.opts.model['img_ctx_dim'] = dims[1]
            self.img_ctx_dim = self.opts.model['img_ctx_dim']
            self._process_image = process_raw_image

            # Set learnable flag
            set_learnable(self.cnn, self.opts.model['cnn_trainable'])
        else:
            # no-op as features are provided as inputs directly
            self._process_image = lambda x: x

    def setup(self, reset_params=True):
        """Sets up NN topology by creating the layers."""

        # Setup CNN
        self.setup_cnn()

        # Create encoder
        self.enc = TextEncoder(input_size=self.opts.model['emb_dim'],
                               hidden_size=self.opts.model['enc_dim'],
                               n_vocab=self.n_src_vocab,
                               cell_type=self.opts.model['enc_type'],
                               dropout_emb=self.opts.model['dropout_emb'],
                               dropout_ctx=self.opts.model['dropout_ctx'],
                               dropout_rnn=self.opts.model['dropout_enc'],
                               num_layers=self.opts.model['n_encoders'])

        # Create Decoder
        self.dec = ConditionalMMDecoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            ctx_size_dict={'img': self.img_ctx_dim, 'txt': self.txt_ctx_dim},
            fusion_type=self.opts.model['fusion_type'],
            tied_emb=self.opts.model['tied_emb'],
            dec_init=self.opts.model['dec_init'],
            att_type=self.opts.model['att_type'],
            att_activ=self.opts.model['att_activ'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout_out'])

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            self.enc.emb.weight = self.dec.emb.weight

        if reset_params:
            self.reset_parameters()

    def load_data(self, split):
        """Loads the requested dataset split."""
        if split not in self.datasets:
            dataset = Multi30kDataset(split=split,
                                      img_mode=self.opts.model['img_mode'],
                                      data_dict=self.opts.data,
                                      vocabs=self.vocabs,
                                      topology=self.topology,
                                      logger=self.logger)
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
        return {
            'image': (self._process_image(data['image']).transpose(0, 1), None),
            'txt': self.enc(data[self.sl]),
        }

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
