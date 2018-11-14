# -*- coding: utf-8 -*-
import logging

from torch import nn
import torch.nn.functional as F

from ..layers import ImageEncoder, TextEncoder, ConditionalMMDecoder
from ..datasets import MultimodalDataset
from .nmt import NMT

logger = logging.getLogger('nmtpytorch')


class AttentiveRawMNMT(NMT):
    """An end-to-end sequence-to-sequence NMT model with visual attention over
    convolutional features using raw image files as input.
    """
    def set_defaults(self):
        # Set parent defaults
        super().set_defaults()
        self.defaults.update({
            'cnn_type': 'resnet50',     # A variant of VGG or ResNet
            'cnn_layer': 'res5c_relu',  # From where to extract features
            'cnn_pretrained': True,     # Should we use pretrained imagenet weights
            'cnn_finetune': None,       # Should we finetune part or all of CNN
            'pool': None,               # ('Avg|Max', kernel_size, stride_size)
            'dropout_img': 0.,          # a 2d dropout over conv features
            'l2_norm': False,           # L2 normalize features
            'l2_norm_dim': -1,           # Which dimension to L2 normalize
            'fusion_type': 'concat',    # Multimodal context fusion (sum|mul|concat)
            'resize': 256,              # resize width, height for images
            'crop': 224,                # center crop size after resize
        })

    def __init__(self, opts):
        super().__init__(opts)

        assert self.opts.model['cnn_layer'] not in ('avgpool', 'fc', 'pool'), \
            "{} given for 'cnn_layer' but it should be a conv layer.".format(
                self.opts.model['cnn_layer'])

    def reset_parameters(self):
        """Initializes learnable weights with kaiming normal."""
        for name, param in self.named_parameters():
            if (param.requires_grad and 'bias' not in name and
                    not name.startswith('cnn')):
                logger.info('  Initializing weights for {}'.format(name))
                nn.init.kaiming_normal_(param.data)

    def setup(self, is_train=True):
        logger.info('Loading CNN')
        cnn_encoder = ImageEncoder(
            cnn_type=self.opts.model['cnn_type'],
            pretrained=self.opts.model['cnn_pretrained'])

        # Set truncation point
        cnn_encoder.setup(layer=self.opts.model['cnn_layer'],
                          dropout=self.opts.model['dropout_img'],
                          pool=self.opts.model['pool'])

        # By default the CNN is not tuneable
        if self.opts.model['cnn_finetune'] is not None:
            assert not self.opts.model['l2_norm'], \
                "finetuning and l2 norm does not work together."
            cnn_encoder.set_requires_grad(
                value=True, layers=self.opts.model['cnn_finetune'])

        # Number of channels defines the spatial vector dim for us
        self.ctx_sizes['image'] = cnn_encoder.get_output_shape()[1]

        # Finally set the CNN as a submodule
        self.cnn = cnn_encoder.get()

        # Nicely printed table of summary for the CNN
        logger.info(cnn_encoder)

        ########################
        # Create Textual Encoder
        ########################
        self.enc = TextEncoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['enc_dim'],
            n_vocab=self.n_src_vocab,
            rnn_type=self.opts.model['enc_type'],
            dropout_emb=self.opts.model['dropout_emb'],
            dropout_ctx=self.opts.model['dropout_ctx'],
            dropout_rnn=self.opts.model['dropout_enc'],
            num_layers=self.opts.model['n_encoders'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'])

        # Create Decoder
        self.dec = ConditionalMMDecoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name=str(self.sl),
            fusion_type=self.opts.model['fusion_type'],
            tied_emb=self.opts.model['tied_emb'],
            dec_init=self.opts.model['dec_init'],
            att_type=self.opts.model['att_type'],
            att_activ=self.opts.model['att_activ'],
            transform_ctx=self.opts.model['att_transform_ctx'],
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout_out'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'])

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            self.enc.emb.weight = self.dec.emb.weight

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

        return {
            'image': (feats, None),
            str(self.sl): self.enc(batch[self.sl]),
        }
