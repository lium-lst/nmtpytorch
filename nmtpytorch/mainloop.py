# -*- coding: utf-8 -*-
import time
import logging

from .evaluator import Evaluator
from .optimizer import Optimizer
from .monitor import Monitor
from .utils.gpu import GPUManager
from .utils.misc import get_module_groups
from .utils.misc import load_pt_file, fix_seed
from .utils.ml_metrics import Loss
from .utils.data import make_dataloader
from .utils.tensorboard import TensorBoard
from .search import beam_search

logger = logging.getLogger('nmtpytorch')


class MainLoop(object):
    def __init__(self, model, train_opts, history=None, mode='train'):
        # Get all training options into this mainloop
        self.__dict__.update(train_opts)

        self.print = logger.info
        self.model = model
        self.epoch_valid = (self.eval_freq == 0)

        # Load training and validation data & create iterators
        self.print('Loading dataset(s)')
        self.train_iterator = make_dataloader(
            self.model.load_data('train', self.batch_size),
            self.pin_memory, self.num_workers)

        # Create monitor for validation, evaluation, checkpointing stuff
        self.monitor = Monitor(self.save_path / self.subfolder, self.exp_id,
                               self.model, logger, self.patience,
                               self.eval_metrics,
                               history=history,
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

        # Move to cuda
        self.model.cuda()
        self.print(self.model)

        # Create optimizer instance
        self.optim = Optimizer(
            self.optimizer, self.model, lr=self.lr, momentum=self.momentum,
            nesterov=self.nesterov, weight_decay=self.l2_reg, gclip=self.gclip)
        self.print(self.optim)

        # Create TensorBoard logger if possible and requested
        self.tb = TensorBoard(self.model, self.tensorboard_dir,
                              self.exp_id, self.subfolder)
        self.print(self.tb)

        # Shift-by-1 and reseed to reproduce batch orders independently
        # from model initialization etc.
        fix_seed(self.seed + 1)

    def train_epoch(self):
        """Trains a full epoch."""
        self.print('Starting Epoch {}'.format(self.monitor.ectr))

        nn_sec = 0.0
        eval_sec = 0.0
        loss_meter = Loss()
        total_sec = time.time()

        for batch in self.train_iterator:
            #############################
            self.monitor.uctr += 1
            nn_start = time.time()

            # Reset gradients
            self.optim.zero_grad()

            # Forward pass returns dict
            batch.to_gpu()
            out = self.model(batch)
            loss_meter.update(out['loss'], out['n_items'])

            # Normalize and sum with auxiliary multi-task losses
            loss = out['loss'] / out['n_items']
            if self.model.aux_loss:
                loss += sum(list(self.model.aux_loss.values()))

            # Backward pass
            loss.backward()

            # Update parameters (includes gradient clipping logic)
            self.optim.step()

            # Keep stats
            nn_sec += (time.time() - nn_start)
            #############################

            if self.monitor.uctr % self.disp_freq == 0:
                # Send statistics
                self.tb.log_scalar(
                    'train_LOSS', loss_meter.batch_loss, self.monitor.uctr)

                msg = "Epoch {} - update {:10d} => loss: {:.3f}".format(
                    self.monitor.ectr, self.monitor.uctr,
                    loss_meter.batch_loss)
                for key, value in self.model.aux_loss.items():
                    val = value.data.cpu()[0]
                    msg += ' [{}: {:.3f}]'.format(key, val)
                    self.tb.log_scalar('train_' + key.upper(),
                                       val, self.monitor.uctr)

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
        epoch_loss = loss_meter.get()
        self.monitor.train_loss.append(epoch_loss)

        self.print("--> Epoch {} finished with mean loss {:.5f}".format(
            self.monitor.ectr, epoch_loss))
        self.print("--> Overhead/Training/Evaluation: {:.2f}/{:.2f}/{:.2f} "
                   "mins (total: {:.2f} mins)   ({} samples/sec)".format(
                       overhead_min, nn_min, eval_min, total_min,
                       int(len(self.train_iterator.dataset) / nn_sec)))
        self.print("Peak memory usage: {}".format(GPUManager.get_mem_usage()))

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
        self.model.train(False)

        # Collect simple validation stats first
        results.extend(self.model.test_performance(self.vloss_iterator))

        if self.monitor.beam_metrics:
            self.print('Performing translation search (beam_size:{})'.format(
                self.eval_beam))
            beam_time = time.time()
            hyps = beam_search([self.model], self.beam_iterator,
                               beam_size=self.eval_beam)
            beam_time = time.time() - beam_time

            # Compute metrics and update results
            score_time = time.time()
            results.extend(self.evaluator.score(hyps))
            score_time = time.time() - score_time

            self.print("Beam Search: {:.2f} sec, Scoring: {:.2f} sec "
                       "({} sent/sec)".format(beam_time, score_time,
                                              int(len(hyps) / beam_time)))

        # Log metrics to tensorboard
        self.tb.log_metrics(results, self.monitor.uctr, suffix='val_')

        # Add new scores to history
        self.monitor.update_scores(results)

        # Check early-stop criteria and save snapshots if any
        self.monitor.save_models()

        # Dump summary and switch back to training mode
        self.monitor.val_summary()
        self.model.train(True)

    def __call__(self):
        """Runs training loop."""
        self.print('Training started on %s' % time.strftime('%d-%m-%Y %H:%M'))
        self.model.train(True)

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

        self.print("Peak memory usage: {}".format(GPUManager.get_mem_usage()))
        self.print('Training finished on %s' % time.strftime('%d-%m-%Y %H:%M'))
        # Close tensorboard
        self.tb.close()
