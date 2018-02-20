import torch
from torch.autograd import Variable
from torchvision import models
from torchvision.models.vgg import cfg as vgg_cfg


def get_vgg_names(config, batch_norm=False):
    names = []

    # Counters for layer naming
    n_block, n_conv = 1, 1

    for v in config:
        if v == 'M':
            names.append('pool%d' % n_block)
            n_block += 1
            n_conv = 1
        else:
            conv_name = 'conv%d_%d' % (n_block, n_conv)
            names.append(conv_name)
            if batch_norm:
                names.append('%s+bn' % conv_name)
                names.append('%s+bn+relu' % conv_name)
            else:
                names.append('%s+relu' % conv_name)

            n_conv += 1

    return names


resnet_layers = ['conv1', 'bn_conv1', 'conv1_relu', 'pool1',
                 'res2c_relu', 'res3d_relu', 'res4f_relu',
                 'res5c_relu', 'avgpool']


class ImageEncoder(object):
    CFG_MAP = {
        'resnet18': resnet_layers,
        'resnet34': resnet_layers,
        'resnet50': resnet_layers,
        'resnet101': resnet_layers,
        'resnet152': resnet_layers,
        # Plain VGGs
        'vgg11': get_vgg_names(vgg_cfg['A']),
        'vgg13': get_vgg_names(vgg_cfg['B']),
        'vgg16': get_vgg_names(vgg_cfg['D']),
        'vgg19': get_vgg_names(vgg_cfg['E']),
        # Batchnorm VGGs
        'vgg11_bn': get_vgg_names(vgg_cfg['A'], batch_norm=True),
        'vgg13_bn': get_vgg_names(vgg_cfg['B'], batch_norm=True),
        'vgg16_bn': get_vgg_names(vgg_cfg['D'], batch_norm=True),
        'vgg19_bn': get_vgg_names(vgg_cfg['E'], batch_norm=True),
    }

    def __init__(self, cnn_type, pretrained=True):
        self.cnn_type = cnn_type
        self.pretrained = pretrained

        assert self.cnn_type in self.CFG_MAP, \
            "{} not supported by our feature extractor".format(self.cnn_type)

        # Load CNN instance
        self.cnn = getattr(models, self.cnn_type)(pretrained=pretrained)

        # Get layer name to index mapping
        self.layer_map = self.CFG_MAP[self.cnn_type]

    def get(self, layer):
        """Returns features from the requested layer with feature dim info"""
        assert layer in self.layer_map, "The given layer is not known."

        if self.cnn_type.startswith('vgg'):
            parameters = list(self.cnn.features)
        elif self.cnn_type.startswith('resnet'):
            parameters = list(self.cnn.children())

        # Get the exact cut-point for the requested layer
        layer_idx = self.layer_map.index(layer)
        cnn = torch.nn.Sequential(*parameters[:layer_idx + 1])

        # Dummy test to detect output number filters
        x = Variable(torch.rand(1, 3, 224, 224), volatile=True)

        # Returns (1, n_channel, w, h)
        size = list(cnn.forward(x).size())
        return (cnn, size)
