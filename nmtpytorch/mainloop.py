# -*- coding: utf-8 -*-
import time
import logging

import torch

from .evaluator import Evaluator
from .optimizer import Optimizer
from .monitor import Monitor
from .utils.misc import get_module_groups
from .utils.misc import load_pt_file, fix_seed
from .utils.ml_metrics import Loss
from .utils.data import make_dataloader
from .utils.tensorboard import TensorBoard
from .search import beam_search

logger = logging.getLogger('nmtpytorch')


class MainLoop:
    def __init__(self, model, train_opts, dev_mgr):
        # Get all training options into this mainloop
        self.__dict__.update(train_opts)

        self.print = logger.info
        self.model = model
        self.dev_mgr = dev_mgr
        self.epoch_valid = (self.eval_freq == 0)
        self.oom_count = 0
        self.loss_meter = Loss()

        # Load training and validation data & create iterators
        self.print('Loading dataset(s)')
        self.train_iterator = make_dataloader(
            self.model.load_data('train', self.batch_size),
            self.pin_memory, self.num_workers)

        # Create monitor for validation, evaluation, checkpointing stuff
        self.monitor = Monitor(self.save_path / self.subfolder, self.exp_id,
                               self.model, logger, self.patience,
                               self.eval_metrics,
                               save_best_metrics=self.save_best_metrics,
                               n_checkpoints=self.n_checkpoints)

        # If a validation set exists
        if 'val_set' in self.model.opts.data and self.eval_freq >= 0:
            if 'LOSS' in self.monitor.eval_metrics:
                self.vloss_iterator = make_dataloader(
                    self.model.load_data('val', self.batch_size, mode='eval'))

            if self.monitor.beam_metrics is not None:
                self.beam_iterator = make_dataloader(
                    self.model.load_data('val', self.eval_batch_size, mode='beam'))
                # Create hypothesis evaluator
                self.evaluator = Evaluator(
                    self.model.val_refs, self.monitor.beam_metrics,
                    filters=self.eval_filters)

        # Setup model
        self.model.setup()
        self.model.reset_parameters()

        ################################################
        # Initialize model weights with a pretrained one
        # This should come after model.setup()
        ################################################
        if train_opts['pretrained_file']:
            # Relax the strict condition for partial initialization
            weights, _, _ = load_pt_file(train_opts['pretrained_file'])
            for name in get_module_groups(weights.keys()):
                self.print(
                    ' -> will initialize {}.* with pretrained weights.'.format(name))
            model.load_state_dict(weights, strict=False)

        ############################
        # Freeze layers if requested
        ############################
        if train_opts['freeze_layers']:
            frozen = []
            for layer in train_opts['freeze_layers'].split(','):
                for name, param in self.model.named_parameters():
                    if name.startswith(layer):
                        param.requires_grad = False
                        frozen.append(name)

            for name in get_module_groups(frozen):
                self.print(' -> froze parameter {}.*'.format(name))

        self.print(self.model)
        self.model = self.model.to(self.dev_mgr.dev)

        if self.dev_mgr.req_cpu or len(self.dev_mgr.cuda_dev_ids) == 1:
            self.net = self.model
        else:
            self.net = torch.nn.DataParallel(
                self.model, device_ids=self.dev_mgr.cuda_dev_ids, dim=1)

        # Create optimizer instance
        self.optim = Optimizer(
            self.optimizer, self.model, lr=self.lr, momentum=self.momentum,
            nesterov=self.nesterov, weight_decay=self.l2_reg,
            gclip=self.gclip, lr_decay=self.lr_decay,
            lr_decay_factor=self.lr_decay_factor,
            lr_decay_mode=self.monitor.lr_decay_mode,
            lr_decay_min=self.lr_decay_min,
            lr_decay_patience=self.lr_decay_patience)
        self.print(self.optim)

        # Create TensorBoard logger if possible and requested
        self.tboard = TensorBoard(self.model, self.tensorboard_dir,
                                  self.exp_id, self.subfolder)
        self.print(self.tboard)

        # Shift-by-1 and reseed to reproduce batch orders independently
        # from model initialization etc.
        fix_seed(self.seed + 1)

    def train_batch(self, batch):
        """Trains a batch."""
        nn_start = time.time()

        # Reset gradients
        self.optim.zero_grad()

        # Forward pass with training progress
        # NOTE: Problematic for multi-gpu
        out = self.net(batch, uctr=self.monitor.uctr, ectr=self.monitor.ectr)
        if 'loss' in out:
            # NOTE: Fix this afterwards so that every model adapts the same style
            # Classical models have single loss
            self.loss_meter.update(out['loss'], out['n_items'])
            loss = out['loss'] / out['n_items']
        else:
            # NOTE: For now, let's simply average losses for MTL
            for tid in out:
                self.loss_meter.update(out[tid]['loss'], out[tid]['n_items'])
            # Normalize the losses and take the average
            # NOTE: averaging may not be a good idea if the model multiplies
            # them with scalar weights
            loss = sum([out[k]['loss'] / out[k]['n_items'] for k in out]) / len(out)

        # Add other losses if any
        if self.net.aux_loss:
            loss += sum(list(self.net.aux_loss.values()))

        # Backward pass
        loss.backward()

        # Update parameters (includes gradient clipping logic)
        self.optim.step()

        return time.time() - nn_start

    def train_epoch(self):
        """Trains a full epoch."""
        self.print('Starting Epoch {}'.format(self.monitor.ectr))

        nn_sec = 0.0
        eval_sec = 0.0
        total_sec = time.time()
        self.loss_meter.reset()
        self.oom_count = 0

        for batch in self.train_iterator:
            batch.device(self.dev_mgr.dev)
            self.monitor.uctr += 1

            try:
                nn_sec += self.train_batch(batch)
            except RuntimeError as e:
                if self.handle_oom and 'out of memory' in e.args[0]:
                    torch.cuda.empty_cache()
                    self.oom_count += 1
                else:
                    raise e

            if self.monitor.uctr % self.disp_freq == 0:
                # Send statistics
                self.tboard.log_scalar(
                    'train_LOSS', self.loss_meter.batch_loss, self.monitor.uctr)

                msg = "Epoch {} - update {:10d} => loss: {:>7.3f}".format(
                    self.monitor.ectr, self.monitor.uctr,
                    self.loss_meter.batch_loss)
                for key, value in self.net.aux_loss.items():
                    val = value.item()
                    msg += ' [{}: {:.3f}]'.format(key, val)
                    self.tboard.log_scalar('train_' + key.upper(), val, self.monitor.uctr)
                msg += ' (#OOM: {})'.format(self.oom_count)
                self.print(msg)

            # Do validation?
            if (not self.epoch_valid and
                    self.monitor.ectr >= self.eval_start and
                    self.eval_freq > 0 and
                    self.monitor.uctr % self.eval_freq == 0):
                eval_start = time.time()
                self.do_validation()
                eval_sec += time.time() - eval_start

            if (self.checkpoint_freq and self.n_checkpoints > 0 and
                    self.monitor.uctr % self.checkpoint_freq == 0):
                self.print('Saving checkpoint...')
                self.monitor.save_checkpoint()

            # Check stopping conditions
            if self.monitor.early_bad == self.monitor.patience:
                self.print("Early stopped.")
                return False

            if self.monitor.uctr == self.max_iterations:
                self.print("Max iterations {} reached.".format(
                    self.max_iterations))
                return False

        # All time spent for this epoch
        total_min = (time.time() - total_sec) / 60
        # All time spent during forward/backward/step
        nn_min = nn_sec / 60
        # All time spent during validation(s)
        eval_min = eval_sec / 60
        # Rest is iteration overhead + checkpoint saving
        overhead_min = total_min - nn_min - eval_min

        # Compute epoch loss
        epoch_loss = self.loss_meter.get()
        self.monitor.train_loss.append(epoch_loss)

        self.print("--> Epoch {} finished with mean loss {:.5f}".format(
            self.monitor.ectr, epoch_loss))
        self.print("--> Overhead/Training/Evaluation: {:.2f}/{:.2f}/{:.2f} "
                   "mins (total: {:.2f} mins)   ({} samples/sec)".format(
                       overhead_min, nn_min, eval_min, total_min,
                       int(len(self.train_iterator.dataset) / nn_sec)))

        # Do validation?
        if self.epoch_valid and self.monitor.ectr >= self.eval_start:
            self.do_validation()

        # Check whether maximum epoch is reached
        if self.monitor.ectr == self.max_epochs:
            self.print("Max epochs {} reached.".format(self.max_epochs))
            return False

        self.monitor.ectr += 1
        return True

    def do_validation(self):
        """Do early-stopping validation."""
        results = []
        self.monitor.vctr += 1
        self.net.train(False)
        torch.set_grad_enabled(False)

        # Collect simple validation stats first
        self.print('Computing evaluation loss...')
        results.extend(self.net.test_performance(self.vloss_iterator))

        if self.monitor.beam_metrics:
            self.print('Performing beam search (beam_size:{})'.format(
                self.eval_beam))
            beam_time = time.time()
            # For multitask learning models, language-specific validation uses
            # by default the 0th Topology in val_tasks
            task = None
            if hasattr(self.net, 'val_tasks'):
                task = self.net.val_tasks[0].direction
            hyps = beam_search([self.net], self.beam_iterator,
                               task_id=task,
                               beam_size=self.eval_beam,
                               max_len=self.eval_max_len)
            beam_time = time.time() - beam_time

            # Compute metrics and update results
            score_time = time.time()
            results.extend(self.evaluator.score(hyps))
            score_time = time.time() - score_time

            self.print("Beam Search: {:.2f} sec, Scoring: {:.2f} sec "
                       "({} sent/sec)".format(beam_time, score_time,
                                              int(len(hyps) / beam_time)))

        # Log metrics to tensorboard
        self.tboard.log_metrics(results, self.monitor.uctr, suffix='val_')

        # Add new scores to history
        self.monitor.update_scores(results)

        # Do a scheduler LR step
        lr_change = self.optim.lr_step(self.monitor.get_last_eval_score())
        if lr_change and self.lr_decay_revert:
            self.print('Reloading previous best model parameters')
            self.monitor.reload_previous_best()

        # Check early-stop criteria and save snapshots if any
        self.monitor.save_models()

        # Dump summary and switch back to training mode
        self.monitor.val_summary()
        self.net.train(True)
        torch.set_grad_enabled(True)

    def __call__(self):
        """Runs training loop."""
        self.print('Training started on %s' % time.strftime('%d-%m-%Y %H:%M:%S'))
        self.net.train(True)
        torch.set_grad_enabled(True)

        # Evaluate once before even starting training
        if self.eval_zero:
            self.do_validation()

        while self.train_epoch():
            pass

        if self.monitor.vctr > 0:
            self.monitor.val_summary()
        else:
            # No validation done, save final model
            self.print('Saving final model.')
            self.monitor.save_model(suffix='final')

        self.print('Training finished on %s' % time.strftime('%d-%m-%Y %H:%M'))
        # Close tensorboard
        self.tboard.close()
