# -*- coding: utf-8 -*-
import math
from collections import OrderedDict
import logging

import torch
from torch import nn

from .amnmtfeats import AttentiveMNMTFeatures

logger = logging.getLogger('nmtpytorch')


class AttentiveMNMTFeaturesFA(AttentiveMNMTFeatures):
    """Filtered attention variant of multimodal attention."""
    def set_defaults(self):
        # Set parent defaults
        super().set_defaults()

    def __init__(self, opts):
        super().__init__(opts)

    def setup(self, is_train=True):
        super().setup(is_train)

        # From 2560(2048+512) to 512
        # To 1*H*W for attention scores
        in_channels = sum(self.ctx_sizes.values())
        self.enc_att = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, self.ctx_sizes[self.sl], 1, 1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(self.ctx_sizes[self.sl], 1, 1, 1)),
            ('relu2', nn.ReLU()),
        ]))

    def encode(self, batch, **kwargs):
        # Get source language encodings (S*B*C)
        text_encoding = self.enc(batch[self.sl])

        # Infer spatial width/height
        w = int(math.sqrt(batch['image'].shape[0]))

        # image features will come as: (HW,B,C) -> (B, C, HW) -> (B, C, H, W)
        conv_map = batch['image'].permute(1, 2, 0).view(
            *batch['image'].shape[1:], w, w)

        # Get last encoding (B*C)
        last_encoding = text_encoding[0][-1]

        # Tile over spatial dimensions: B*C*H*W
        tiled_encoding = last_encoding[..., None, None].expand(
            -1, -1, *conv_map.shape[2:])

        # Final concat representation: B*(D+C)*H*W
        concat = torch.cat([conv_map, tiled_encoding], dim=1)

        att_scores = self.enc_att(concat)
        att_probs = nn.functional.softmax(
            att_scores.view(batch.size, -1), dim=1).view_as(att_scores)

        # Filter features
        feats = conv_map * att_probs

        # Get features into (B, C, HW) and then (HW, B, C)
        feats = feats.view(*feats.shape[:2], -1).permute(2, 0, 1)

        return {
            str(self.sl): text_encoding,
            'image': (feats, None),
        }
