# -*- coding: utf-8 -*-
from ..layers import ImageEncoder, XuDecoder

from ..datasets import Multi30kRawDataset

from .nmt import NMT


class ShowAttendAndTell(NMT):
    r"""An Implementation of 'Show, attend and tell' image captioning paper.

    Paper: http://www.jmlr.org/proceedings/papers/v37/xuc15.pdf
    Reference implementation: https://github.com/kelvinxu/arctic-captions
    """
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            'emb_dim': 128,             # Source and target embedding sizes
            'emb_maxnorm': None,        # Normalize embeddings l2 norm to 1
            'emb_gradscale': False,     # Scale embedding gradients w.r.t. batch frequency
            'dec_dim': 256,             # Decoder hidden size
            'dec_type': 'gru',          # Decoder type (gru|lstm)
            'dec_init': 'mean_ctx',     # How to initialize decoder (zero/mean_ctx)
            'trg_bos': 'emb',           # emb: Learn a <bos> and use it
                                        # ctx: Source driven dynamic <bos>
            'att_type': 'mlp',          # Attention type (mlp|dot)
            'att_temp': 1.,             # Attention temperature
            'att_activ': 'tanh',        # Attention non-linearity (all torch nonlins)
            'att_mlp_bias': False,      # Enables bias in attention mechanism
            'att_bottleneck': 'ctx',    # Bottleneck dimensionality (ctx|hid)
            'att_transform_ctx': True,  # Transform annotations before attention
            'dropout': 0,               # Simple dropout
            'tied_emb': False,          # Share embeddings: (False|2way|3way)
            'direction': None,          # Network directionality, i.e. en->de
            'selector': False,          # Selector gate
            'prev2out': True,           # Add prev embedding to output
            'ctx2out': True,            # Add context to output
            'cnn_type': 'resnet50',     # A variant of VGG or ResNet
            'cnn_layer': 'res5c_relu',  # From where to extract features
            'cnn_pretrained': True,     # Should we use pretrained imagenet weights
            'cnn_finetune': None,       # Should we finetune part or all of CNN
            'pool': None,               # ('Avg|Max', kernel_size, stride_size)
            'resize': 256,              # resize width, height for images
            'crop': 224,                # center crop size after resize
        }

    def __init__(self, opts, logger=None):
        super().__init__(opts, logger)

    def setup(self, is_train=True):
        self.print('Loading CNN')
        cnn_encoder = ImageEncoder(
            cnn_type=self.opts.model['cnn_type'],
            pretrained=self.opts.model['cnn_pretrained'])

        # Set truncation point
        cnn_encoder.setup(
            layer=self.opts.model['cnn_layer'], pool=self.opts.model['pool'])

        # By default the CNN is not tuneable
        if self.opts.model['cnn_finetune'] is not None:
            cnn_encoder.set_requires_grad(
                value=True, layers=self.opts.model['cnn_finetune'])

        # Number of channels defines the spatial vector dim for us
        self.ctx_sizes = {'image': cnn_encoder.get_output_shape()[1]}

        # Finally set the CNN as a submodule
        self.cnn = cnn_encoder.get()

        # Nicely printed table of summary for the CNN
        self.print(cnn_encoder)

        # Create Decoder
        self.dec = XuDecoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name='image',
            tied_emb=self.opts.model['tied_emb'],
            dec_init=self.opts.model['dec_init'],
            att_type=self.opts.model['att_type'],
            att_temp=self.opts.model['att_temp'],
            att_activ=self.opts.model['att_activ'],
            transform_ctx=self.opts.model['att_transform_ctx'],
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout=self.opts.model['dropout'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            selector=self.opts.model['selector'],
            prev2out=self.opts.model['prev2out'],
            ctx2out=self.opts.model['ctx2out'])

    def load_data(self, split):
        """Loads the requested dataset split."""
        self.datasets[split] = Multi30kRawDataset(
            data_dict=self.opts.data[split + '_set'],
            warmup=(split != 'train'),
            resize=self.opts.model['resize'],
            crop=self.opts.model['crop'],
            vocabs=self.vocabs,
            topology=self.topology)
        self.print(self.datasets[split])

    def encode(self, batch):
        # Get features into (n,c,w*h) and then (w*h,n,c)
        feats = self.cnn(batch['image'])
        feats = feats.view((*feats.shape[:2], -1)).permute(2, 0, 1)

        return {'image': (feats, None)}

    # def f_next(self, img_ctx, y_t, c_t, h_t):
        # """Defines the inner block of recurrence."""
        # # Apply attention
        # alpha_t, z_t = self.ff_att(img_ctx, h_t)

        # # Get next state
        # h_t, c_t = self.decoder(dec_inp, (h_t, c_t))
        # if self.opts.model['dropout']:
            # h_t = self.dropout_lstm(h_t)

        # # This h_t, (optionally along with embs and z_t)
        # # will connect to softmax() predictions.
        # logit = self.ff_out_lstm(h_t)

        # if self.opts.model['prev2out']:
            # logit += y_t

        # if self.opts.model['ctx2out']:
            # logit += self.ff_out_ctx(z_t)

        # # Unnormalized vocabulary scores
        # logit = F.tanh(logit)
        # if self.opts.model['dropout']:
            # logit = self.dropout_logit(logit)

        # # Compute softmax
        # log_p = F.log_softmax(self.ff_pre_softmax(logit), dim=1)

        # return log_p, c_t, h_t, alpha_t
