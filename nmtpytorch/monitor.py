# -*- coding: utf-8 -*-
from pathlib import Path
from collections import OrderedDict, deque

import numpy as np
import torch

from .utils.io import FileRotator


class Monitor(object):
    # Variables to save
    VARS = ['uctr', 'ectr', 'vctr', 'early_bad', 'train_loss', 'val_scores']

    def __init__(self, save_path, exp_id,
                 model, logger, patience, eval_metrics, history=None,
                 save_best_metrics=False, n_checkpoints=0):
        self.print = logger.info
        self.save_path = save_path
        self.exp_id = exp_id
        self.model = model
        self.patience = patience
        # From now on, metric names will propagate as upper-case
        self.eval_metrics = eval_metrics.upper().split(',')
        self.save_best_metrics = save_best_metrics
        self.checkpoints = FileRotator(n_checkpoints)

        if history is None:
            history = {}

        self.uctr = history.pop('uctr', 0)            # update ctr
        self.ectr = history.pop('ectr', 1)            # epoch ctr
        self.vctr = history.pop('vctr', 0)            # validation ctr
        self.early_bad = history.pop('early_bad', 0)  # earlystop patience
        self.train_loss = history.pop('train_loss', [])
        self.val_scores = history.pop('val_scores', OrderedDict())

        if len(self.eval_metrics) > 1:
            # To keep current best metric validation id and score
            self.cur_bests = {}

            # first one is early-stop metric
            self.early_metric = self.eval_metrics[0]

            # First metric is considered to be early-stopping metric
            # Copy eval_metrics and remove 'loss'
            self.beam_metrics = self.eval_metrics[:]
            self.beam_metrics.remove('LOSS')

            # Initialize with empty lists if not available
            for metric in self.eval_metrics:
                if metric not in self.val_scores:
                    self.val_scores[metric] = []

    @staticmethod
    def best_score(scores):
        """Returns the best validation id and score for that."""
        idx, score = sorted(enumerate(scores), key=lambda e: e[1],
                            reverse=scores[0].higher_better)[0]
        return (idx + 1, score)

    def state_dict(self):
        """Returns a dictionary of stateful variables."""
        return {k: getattr(self, k) for k in self.VARS}

    def val_summary(self):
        """Prints a summary of validation results."""
        self.print('--> This is model: {}'.format(self.exp_id))
        for name, (vctr, score) in self.cur_bests.items():
            self.print('--> Best {} so far: {:.2f} @ validation {}'.format(
                name, score.score, vctr))

    def _save(self, fname):
        """Saves the current model on disk."""
        torch.save(
            {
                'opts': self.model.opts.to_dict(),
                'model': self.model.state_dict(),
                'history': self.state_dict(),
            }, fname)

    def save_checkpoint(self):
        """Saves a checkpoint by keeping track of file rotation."""
        self.checkpoints.push(
            self.save_model(suffix='update{}'.format(self.uctr)))

    def save_model(self, metric=None, suffix='', do_symlink=False):
        """Saves a checkpoint with arbitrary suffix(es) appended."""
        # Construct file name
        fname = self.exp_id
        if metric:
            self.print('Saving best model based on {}'.format(metric.name))
            fname += "-val{:03d}.best.{}_{:.3f}".format(
                self.vctr, metric.name.lower(), metric.score)
        if suffix:
            fname += "-{}".format(suffix)
        fname = self.save_path / (fname + ".ckpt")

        # Save the file
        self._save(fname)

        # Also create a symbolic link to the above checkpoint for the metric
        if metric and do_symlink:
            symlink = "{}.best.{}.ckpt".format(self.exp_id, metric.name.lower())
            symlink = self.save_path / symlink
            if symlink.exists():
                old_ckpt = symlink.resolve()
                symlink.unlink()
                old_ckpt.unlink()
            symlink.symlink_to(fname.name)

        return fname

    def update_scores(self, results):
        """Updates score lists and current bests."""
        for metric in results:
            self.print('Validation {} -> {}'.format(self.vctr, metric))
            self.val_scores[metric.name].append(metric)
            self.cur_bests[metric.name] = self.best_score(
                self.val_scores[metric.name])

    def save_models(self):
        cur_bests = self.cur_bests.copy()

        # Let's start with early-stopping metric
        vctr, metric = cur_bests.pop(self.early_metric)
        if vctr == self.vctr:
            self.early_bad = 0
            self.save_model(metric=metric, do_symlink=True)
        else:
            # Increment counter
            self.early_bad += 1

        # If requested, save all best metric snapshots
        if self.save_best_metrics and cur_bests:
            for (vctr, metric) in cur_bests.values():
                if vctr == self.vctr:
                    self.save_model(metric=metric, do_symlink=True)

        self.print('Early stopping patience: {}'.format(
            self.patience - self.early_bad))
