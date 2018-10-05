# -*- coding: utf-8 -*-
import torch
from torch import nn

# Layer contributed by @elliottd


class MaxMargin(nn.Module):
    """A max-margin layer for ranking-based loss functions."""

    def __init__(self, margin, max_violation=False):
        super().__init__()

        assert margin > 0., "margin must be > 0."

        # Other arguments
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, enc1, enc2):
        """Computes the max-margin loss given a pair of rank-2
           annotation matrices. The matrices must have the same number of
           batches and the same number of feats.

        Arguments:
            enc1(Tensor): A tensor of `B*feats` representing the
                annotation vectors of the first encoder.
            enc2(Tensor): A tensor of `B*feats` representation the
                annotation vectors of the second encoder.
        """

        assert enc1.shape == enc2.shape, \
            "shapes must match: enc1 {} enc2 {}".format(enc1.shape, enc2.shape)

        enc1 = enc1 / enc1.norm(p=2, dim=1).unsqueeze(1)
        enc2 = enc2 / enc2.norm(p=2, dim=1).unsqueeze(1)
        loss = self.constrastive_loss(enc1, enc2)

        return {'loss': loss}

    def constrastive_loss(self, enc1, enc2):
        if enc1.shape[0] == 1:
            # There is no error when we have a single-instance batch.
            # Return a dummy error of 1e-5 as a regularizer
            return torch.tensor([1e-3], device=enc1.device)

        # compute enc1-enc2 score matrix
        scores = self.cosine_sim(enc1, enc2)
        diagonal = scores.diag().view(enc1.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_enc1 = (self.margin + scores - d2).clamp(min=0)
        cost_enc2 = (self.margin + scores - d1).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0), device=enc1.device) > .5
        cost_enc2 = cost_enc2.masked_fill_(mask, 0)
        cost_enc1 = cost_enc1.masked_fill_(mask, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_enc2 = cost_enc2.max(1)[0]
            cost_enc1 = cost_enc1.max(0)[0]
        denom = cost_enc1.shape[0]**2 - cost_enc1.shape[0]
        return (cost_enc2 + cost_enc1).sum() / denom

    def cosine_sim(self, one, two):
        '''Cosine similarity between all the first and second encoder pairs'''
        return one.mm(two.t())
