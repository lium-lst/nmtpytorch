from collections import defaultdict

import numpy as np
import torch

from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_average_precision_score as lrap

from ignite import metrics as ig_metrics

from .device import DeviceManager

from ..metrics import Metric


class Loss:
    """Accumulates and computes correctly training and validation losses."""
    def __init__(self):
        self.reset()

    def reset(self):
        self._loss = 0
        self._denom = 0
        self.batch_loss = 0

    def update(self, loss, n_items):
        # Store last batch loss
        self.batch_loss = loss.item()
        # Add it to cumulative loss
        self._loss += self.batch_loss
        # Normalize batch loss w.r.t n_items
        self.batch_loss /= n_items
        # Accumulate n_items inside the denominator
        self._denom += n_items

    def get(self):
        if self._denom == 0:
            return 0
        return self._loss / self._denom

    @property
    def denom(self):
        return self._denom


class Precision:
    """Wrapper metric around `pytorch-ignite`."""
    def __init__(self, is_multilabel=True):
        self.is_multilabel = is_multilabel
        self.__metric = ig_metrics.Precision(
            average=True, is_multilabel=self.is_multilabel)

    def update(self, y_pred, y):
        """Tensors should have N x n_labels dimensions with N = batch_size."""
        self.__metric.update((y_pred, y))

    def compute(self):
        """Once the updates are over, this returns the actual metric."""
        val = 100 * self.__metric.compute()
        return Metric('PRECISION', val, higher_better=True)


class Recall:
    """Wrapper metric around `pytorch-ignite`."""
    def __init__(self, is_multilabel=True):
        self.is_multilabel = is_multilabel
        self.__metric = ig_metrics.Recall(
            average=True, is_multilabel=self.is_multilabel)

    def update(self, y_pred, y):
        """Tensors should have N x n_labels dimensions with N = batch_size."""
        self.__metric.update((y_pred, y))

    def compute(self):
        """Once the updates are over, this returns the actual metric."""
        val = 100 * self.__metric.compute()
        return Metric('RECALL', val, higher_better=True)


class F1:
    """Wrapper metric around `pytorch-ignite`."""
    def __init__(self, is_multilabel=True):
        self.is_multilabel = is_multilabel
        # Create underlying metrics
        self.__precision = ig_metrics.Precision(
            average=False, is_multilabel=self.is_multilabel)

        self.__recall = ig_metrics.Recall(
            average=False, is_multilabel=self.is_multilabel)

        num = self.__precision * self.__recall * 2
        denom = self.__precision + self.__recall + 1e-20
        f1 = num / denom
        self.__metric = ig_metrics.MetricsLambda(
            lambda t: t.mean().item(), f1)

    def update(self, y_pred, y):
        """Tensors should have N x n_labels dimensions with N = batch_size."""
        self.__precision.update((y_pred, y))
        self.__recall.update((y_pred, y))

    def compute(self):
        val = 100 * self.__metric.compute()
        return Metric('F1', val, higher_better=True)


class CoverageError:
    def __init__(self):
        self._cov = 0
        self._n_items = 0

    def update(self, y_true, y_pred):
        self._cov += coverage_error(y_true, y_pred) * y_pred.shape[0]
        self._n_items += y_pred.shape[0]

    def get(self):
        return self._cov / self._n_items


class LRAPScore:
    def __init__(self):
        self._lrap = 0
        self._n_items = 0

    def update(self, y_true, y_pred):
        self._lrap += lrap(y_true, y_pred) * y_pred.shape[0]
        self._n_items += y_pred.shape[0]

    def get(self):
        return self._lrap / self._n_items


class MeanReciprocalRank:
    """Computes the mean reciprocal rank (MRR) metric for a batch along with
    per time-step MRR statistics that accumulate."""
    def __init__(self, n_classes):
        self.denom = torch.arange(1, 1 + n_classes, device=DeviceManager.DEVICE, dtype=torch.float)
        self._mrr_per_timestep = defaultdict(float)
        self._per_timestep_counts = defaultdict(int)

    def update(self, y_true, y_pred):
        # y_pred: tstep x bsize x n_classes
        # y_true: tstep x bsize

        # Get a clone to mask out zero padded elements
        y_true_nz = y_true.clone()
        y_true_nz[y_true_nz.eq(0)] = -1
        y_true_nz.unsqueeze_(-1)

        # Sort negative log-probabilities from most-likely to less-likely
        sorted_logp, sorted_idxs = torch.sort(y_pred, dim=-1, descending=True)

        # matches: tstep x bsize x vocab of binary indicators to mark matches
        matches = (sorted_idxs == y_true_nz).float()

        samples_per_timestep = (y_true > 1).sum(1).tolist()

        # Compute MRR per timestep
        for tstep, n_samples in enumerate(samples_per_timestep):
            self._mrr_per_timestep[tstep + 1] += (
                matches[tstep].sum(0) / self.denom).sum()
            self._per_timestep_counts[tstep + 1] += n_samples

    def normalized_mrr(self):
        x, y = self.per_timestep_mrr()
        return 100. * (x.sum() / y.sum())

    def per_timestep_mrr(self):
        timesteps = list(range(1, 1 + len(self._per_timestep_counts)))
        counts = np.array([self._per_timestep_counts[t] for t in timesteps])
        scores = np.array([self._mrr_per_timestep[t] for t in timesteps])
        return scores, counts
