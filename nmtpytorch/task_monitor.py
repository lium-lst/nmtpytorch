from collections import defaultdict, OrderedDict

import numpy as np
import torch

from .utils.io import FileRotator
from .utils.filterchain import FilterChain
from .utils.misc import get_language
from .utils.ml_metrics import Loss
from .utils.exceptions import EarlyStoppedError
from .metrics import metric_info
from . import metrics as available_metrics
from .logger import Logger

log = Logger()


class HypothesisEvaluator:
    r"""Class to evaluate held-out set hypotheses w.r.t a given set of
    references.

    Arguments:
        refs (str or list): A single filename or a list of filenames
            representing the reference sentences.
        metrics(list): List of metric names to compute after generating
            the hypotheses.
        filters(list, optional): List of post-processing filters
            to be applied to both hypotheses and references in the given order.
            Check :class:`FilterChain` for the list of filters.

    """
    def __init__(self, refs, metrics, filters=None):
        self.refs = list(refs.parent.glob(refs.name))
        assert len(self.refs) > 0, "The reference file(s) can not be found."

        self.kwargs = {}
        self.scorers = OrderedDict()
        self.language = get_language(self.refs[0])
        if self.language is None:
            # Only relevant for METEOR right now
            log.log(f'Could not detect language from file {self.refs[0]!r}, defaulting to en')
            self.language = 'en'

        self.filter = lambda s: s
        if filters is not None:
            self.filter = FilterChain(filters)
            self.refs = self.filter(refs)

        for metric in sorted(metrics):
            self.kwargs[metric] = {'language': self.language}
            self.scorers[metric] = getattr(
                available_metrics, metric.upper() + 'Scorer')()

    def score(self, hyps):
        """Returns a dict of metrics for the given list of hypotheses."""
        assert isinstance(hyps, list), "hyps should be a list."

        # Post-process if requested
        hyps = self.filter(hyps)

        results = []
        for key, scorer in self.scorers.items():
            results.append(
                scorer.compute(self.refs, hyps, **self.kwargs[key]))
        return results


class TaskMonitor:
    """Class that tracks evaluation progress during training. The following
    informations are kept as object attributes:
        self.ectr:       # of epochs done so far
        self.uctr:       # of updates, i.e. mini-batches done so far
        self.vctr:       # of evaluations done on val_set so far
        self.early_bad:  # of consecutive evaluations where the model did not improve
        self.train_loss: List of training losses
        self.val_scores: Dict of lists keeping tracking of validation metrics
    """
    # Variables to save
    VARS = ['uctr', 'ectr', 'vctr', 'early_bad', 'train_loss', 'val_scores']

    def __init__(self, model, task_name, save_path, exp_id, patience,
                 early_metric, beam_size, beam_max_len,
                 eval_freq, other_metrics, references, disp_freq=30,
                 filters=None, n_checkpoints=0,
                 checkpoint_freq=1000, save_best_metrics=False,
                 save_optim_state=False):
        self.model = model
        self.task_name = task_name
        self.save_path = save_path
        self.exp_id = f'{exp_id}.{self.task_name}'
        self.patience = patience
        self.save_best_metrics = save_best_metrics
        self.save_optim_state = save_optim_state
        self.n_checkpoints = n_checkpoints
        self.checkpoints = FileRotator(n_checkpoints)
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
        self.eval_at_epochs = self.eval_freq == 0
        self.disp_freq = disp_freq
        self.beam_size = beam_size
        self.beam_max_len = beam_max_len
        self.aux_train_losses = {}
        self.train_loss = Loss()
        self.val_loss = Loss()
        self.epoch_losses = []
        self.batch_timings = []
        self._tboard = None

        # metric to be used for early-stopping (single metric)
        self.early_metric = early_metric
        assert self.early_metric and isinstance(self.early_metric, str), \
            (f"`evaluation[{self.task_name!r}]['early_metric'] should be"
             " a valid lowercase string")

        # list of other metrics to track
        self.other_metrics = other_metrics
        assert isinstance(self.other_metrics, list), \
            (f"`evaluation[{self.task_name!r}]['other_metrics'] should be"
             " a list of lowercase strings")

        self.all_metrics = list(set([self.early_metric] + self.other_metrics))
        assert self.all_metrics, \
            f"No metrics provided in `evaluation[{self.task_name!r}]` section."

        self.beam_metrics = list(
            filter(lambda m: metric_info[m]['beam_metric'], self.all_metrics))

        self.do_compute_val_loss = 'loss' in self.all_metrics

        if self.beam_metrics:
            self.do_score_hypotheses = True
            # Create an evaluator for beam search results
            self.evaluator = HypothesisEvaluator(
                refs=references, metrics=self.beam_metrics, filters=filters)
        else:
            self.do_score_hypotheses = False

        self._uctr = 0
        self._ectr = 0
        self._vctr = 0
        self._early_bad = 0
        self.val_scores = defaultdict(list)

        # To keep current best metric validation id and score
        self.cur_bests = {}

    def register_tensorboard(self, handle):
        """Registers the tensorboard handle for further usage from monitor."""
        self._tboard = handle

    @property
    def uctr(self):
        """Returns the training iterations counter."""
        return self._uctr

    @uctr.setter
    def uctr(self, value):
        """Update the iteration count and potentially perform periodic stuff."""
        self._uctr = value
        if self._uctr % self.disp_freq == 0:
            # show log progress
            self.log_progress()
        if not self.eval_at_epochs and self._uctr % self.eval_freq == 0:
            # do evaluation
            self.start_validation()
        if self.n_checkpoints > 0 and self._uctr % self.checkpoint_freq == 0:
            self.save_checkpoint()

    @property
    def vctr(self):
        """Returns the validation counter."""
        return self._vctr

    @vctr.setter
    def vctr(self, value):
        self._vctr = value

    @property
    def ectr(self):
        """Returns the training epoch counter."""
        return self._ectr

    @ectr.setter
    def ectr(self, value):
        if value > self._ectr:
            # Epoch changed
            self.epoch_summary()
            if self.eval_at_epochs:
                self.start_validation()
        self._ectr = value

    @staticmethod
    def best_score(scores):
        """Returns the best validation idx and the obtained score."""
        do_reverse = scores[0].higher_better
        idx, score = sorted(
            enumerate(scores), key=lambda e: e[1], reverse=do_reverse)[0]
        return (idx + 1, score)

    def get_last_score(self):
        """Returns the last score for the early-stopping metric."""
        return self.cur_bests[self.early_metric][-1].score

    def start_validation(self):
        """Performs validation loss and optionally beam search."""
        self.model.train(False)
        torch.set_grad_enabled(False)

        ##################
        # Validation logic
        ##################
        log.log_prefix(f'Starting evaluation', self.task_name)
        self.vctr += 1
        results = []

        if self.do_compute_val_loss:
            log.log_prefix(f'Computing validation loss', self.task_name)
            dataset = self.model.iterators[self.task_name]['val.loss']
            results.extend(self.model.test_performance(dataset, self.task_name))

        if self.do_score_hypotheses:
            log.log_prefix('Performing beam search', self.task_name)
            dataset = self.model.iterators[self.task_name]['val.beam']

            hyps = self.model.beam_search(
                [self.model], dataset, beam_size=self.beam_size,
                max_len=self.beam_max_len)

            # Compute metrics and update results
            results.extend(self.evaluator.score(hyps))

        # Log metrics to tensorboard
        self._tboard.log_metrics(results, self.uctr, suffix='val_')

        # Add new scores to history
        for metric in results:
            log.log_prefix(f'Validation {self.vctr} -> {metric}', self.task_name)
            self.val_scores[metric.name.lower()].append(metric)
            self.cur_bests[metric.name.lower()] = self.best_score(
                self.val_scores[metric.name.lower()])

        # Do a scheduler LR step
        lr_change = self.model.optimizers[self.task_name].lr_step(
            self.get_last_score())

        # FIXME: this is problematic for multiple tasks/params/optims etc
        # if lr_change and self.opts.train['lr_decay_revert']:
            # log.log('Reloading previous best model parameters')
            # monitor.reload_previous_best()

        # Check early-stop criteria and save snapshots if any
        self.save_models()
        self.dump_scores()

        self.model.train(True)
        torch.set_grad_enabled(True)

        if self._early_bad == self.patience:
            raise EarlyStoppedError()

    def state_dict(self):
        """Returns a dictionary of stateful variables."""
        return {k: getattr(self, k) for k in self.VARS}

    def dump_scores(self):
        # Dump summary and switch back to training mode
        for name, (vctr, score) in self.cur_bests.items():
            log.log_prefix(
                f'Best {name.upper()} so far: {score.score:.2f} @ validation {vctr}', self.task_name)

    def epoch_summary(self):
        """Prints statistics about the finished epoch."""
        batch_timings = np.array(self.batch_timings)
        mean_batch_time = f'{1000 * batch_timings.mean():.2f} ms/batch'
        epoch_time = f'{batch_timings.sum() / 60:.2f} mins'
        target_wps = self.train_loss.denom / batch_timings.sum()

        epoch_loss = self.train_loss.get()
        self.epoch_losses.append(epoch_loss)

        msg = f'Epoch {self.ectr} ended in {epoch_time} with loss => {epoch_loss:.3f}'
        msg += f' ({mean_batch_time} -- {target_wps:.1f} tok/sec)'
        log.log_prefix(msg, self.task_name)

        self.train_loss.reset()
        self.batch_timings = []

    def log_progress(self):
        """Shows the periodic training progress information."""
        msg = f"Epoch {self.ectr} - iter {self.uctr:10d} => loss: {self.train_loss.batch_loss:>7.3f}"
        log.log_prefix(msg, self.task_name)

        # Log to tensorboard (with auxiliary losses if exist)
        self._tboard.log_scalar(f'{self.task_name}.train_loss',
                                self.train_loss.batch_loss, self.uctr)
        for name, value in self.aux_train_losses.items():
            self._tboard.log_scalar(f'{self.task_name}.train_{name}',
                                    value, self.uctr)

    def save_checkpoint(self):
        """Saves a checkpoint by keeping track of file rotation."""
        log.log_prefix("Saving periodic checkpoint...", self.task_name)
        self.checkpoints.push(
            self.save_model(suffix='update{}'.format(self.uctr)))

    def save_model(self, metric=None, suffix='', do_symlink=False):
        """Saves a checkpoint with arbitrary suffix(es) appended."""
        # Construct file name
        fname = self.exp_id.replace(':', '_')
        if metric:
            log.log_prefix(f'Saving best model based on {metric.name}', self.task_name)
            fname += f"-val{self.vctr:03d}.best.{metric.name.lower()}_{metric.score:.3f}"
        if suffix:
            fname += f"-{suffix}"
        fname = self.save_path / (fname + ".ckpt")

        # Save the file
        model_dict = {
            'opts': self.model.opts.state,
            'model': self.model.state_dict(),
            # FIXME: let's disable history for now
            #'history': self.state_dict(),
        }

        # Add optimizer states
        if self.save_optim_state:
            model_dict['optimizer'] = self.model.optimizers[self.task_name].state_dict()

        torch.save(model_dict, fname)

        # Also create a symbolic link to the above checkpoint for the metric
        if do_symlink and metric:
            symlink = "{}.best.{}.ckpt".format(self.exp_id, metric.name.lower())
            symlink = self.save_path / symlink
            if symlink.exists():
                old_ckpt = symlink.resolve()
                symlink.unlink()
                old_ckpt.unlink()
            symlink.symlink_to(fname.name)

        return fname

    def save_models(self):
        cur_bests = self.cur_bests.copy()

        # Let's start with early-stopping metric
        vctr, metric = cur_bests.pop(self.early_metric)
        if vctr == self.vctr:
            self._early_bad = 0
            self.save_model(metric=metric, do_symlink=True)

            # If requested, save all best metric snapshots
            if self.save_best_metrics and cur_bests:
                for (vctr, metric) in cur_bests.values():
                    self.save_model(metric=metric, do_symlink=True)
        else:
            # Increment counter
            self._early_bad += 1

    def reload_previous_best(self):
        raise NotImplementedError()

    def __repr__(self):
        return f"""\
{self.__class__.__name__}(task={self.task_name!r}, early_stop={self.early_metric!r}, \
other_metrics={self.other_metrics!r},
{' '*len(self.__class__.__name__)} patience={self.patience!r}, do_score_hypotheses={self.do_score_hypotheses!r})"""
