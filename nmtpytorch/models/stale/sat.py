# -*- coding: utf-8 -*-
import logging

import torch
import torch.nn.functional as F
from ..layers import ImageEncoder, XuDecoder

from ..datasets import MultimodalDataset

from .nmt import NMT

logger = logging.getLogger('nmtpytorch')


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
            'att_type': 'mlp',          # Attention type (mlp|dot)
            'att_temp': 1.,             # Attention temperature
            'att_activ': 'tanh',        # Attention non-linearity (all torch nonlins)
            'att_mlp_bias': True,       # Enables bias in attention mechanism
            'att_bottleneck': 'ctx',    # Bottleneck dimensionality (ctx|hid)
            'att_transform_ctx': True,  # Transform annotations before attention
            'dropout': 0,               # Simple dropout
            'tied_emb': False,          # Share embeddings: (False|2way|3way)
            'selector': True,           # Selector gate
            'alpha_c': 0.0,             # Attention regularization
            'prev2out': True,           # Add prev embedding to output
            'ctx2out': True,            # Add context to output
            'cnn_type': 'resnet50',     # A variant of VGG or ResNet
            'cnn_layer': 'res5c_relu',  # From where to extract features
            'cnn_pretrained': True,     # Should we use pretrained imagenet weights
            'cnn_finetune': None,       # Should we finetune part or all of CNN
            'pool': None,               # ('Avg|Max', kernel_size, stride_size)
            'l2_norm': False,           # L2 normalize features
            'l2_norm_dim': -1,          # Which dimension to L2 normalize
            'resize': 256,              # resize width, height for images
            'crop': 224,                # center crop size after resize
            'replicate': 1,             # number of captions/image
            'direction': None,          # Network directionality, i.e. en->de
            'bucket_by': None,          # A key like 'en' to define w.r.t which dataset
                                        # the batches will be sorted
        }

    def __init__(self, opts):
        super().__init__(opts)
        if self.opts.model['alpha_c'] > 0:
            self.aux_loss['alpha_reg'] = 0.0

    def setup(self, is_train=True):
        logger.info('Loading CNN')
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
        logger.info(cnn_encoder)

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

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        dataset = MultimodalDataset(
            data=self.opts.data[split + '_set'],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model.get('max_len', None),
            warmup=(split != 'train'),
            resize=self.opts.model['resize'],
            replicate=self.opts.model['replicate'] if split == 'train' else 1,
            crop=self.opts.model['crop'])
        logger.info(dataset)
        return dataset

    def encode(self, batch, **kwargs):
        # Get features into (n,c,w*h) and then (w*h,n,c)
        feats = self.cnn(batch['image'])
        feats = feats.view((*feats.shape[:2], -1)).permute(2, 0, 1)
        if self.opts.model['l2_norm']:
            feats = F.normalize(
                feats, dim=self.opts.model['l2_norm_dim']).detach()

        return {'image': (feats, None)}

    def forward(self, batch, **kwargs):
        result = super().forward(batch)

        if self.training and self.opts.model['alpha_c'] > 0:
            alpha_loss = (
                1 - torch.cat(self.dec.history['alpha_img']).sum(0)).pow(2).sum(0)
            self.aux_loss['alpha_reg'] = alpha_loss.mean().mul(
                self.opts.model['alpha_c'])

        return result
