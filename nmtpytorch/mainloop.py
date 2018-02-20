# -*- coding: utf-8 -*-
import time

import numpy as np

from .metrics import Metric
from .evaluator import Evaluator
from .optimizer import Optimizer
from .monitor import Monitor
from .utils.misc import get_n_params
from .utils.data import to_var
from .utils.tensorboard import TensorBoard
from .search import beam_search


class MainLoop(object):
    def __init__(self, model, logger, train_opts,
                 history=None, weights=None, mode='train'):
        # Get all training options into this mainloop
        self.__dict__.update(train_opts)

        self.print = logger.info
        self.model = model
        self.mode = mode    # (train|resume)
        self.epoch_valid = (self.eval_freq == 0)

        # Load training and validation data & create iterators
        self.print('Loading dataset(s)')
        self.model.load_data('train')
        self.train_iterator = self.model.datasets['train'].get_iterator(
            self.batch_size)

        # Create monitor for validation, evaluation, checkpointing stuff
        self.monitor = Monitor(self.save_path / self.subfolder, self.exp_id,
                               self.model, logger, self.patience,
                               self.eval_metrics,
                               history=history,
                               save_best_metrics=self.save_best_metrics,
                               n_checkpoints=self.n_checkpoints)

        # If a validation set exists
        if 'val_set' in self.model.opts.data and self.eval_freq >= 0:
            self.model.load_data('val')
            val_set = self.model.datasets['val']
            if 'LOSS' in self.monitor.eval_metrics:
                self.vloss_iterator = val_set.get_iterator(self.batch_size)
            if self.monitor.beam_metrics:
                self.beam_iterator = val_set.get_iterator(
                    self.eval_batch_size, only_source=True)
                # Create hypothesis evaluator
                self.evaluator = Evaluator(
                    self.model.val_refs, self.monitor.beam_metrics,
                    filters=self.eval_filters)

        # Setup model
        self.model.setup()

        # This should come after model.setup()
        if weights is not None:
            self.print('Loading pretrained model weights')
            # If not resuming -> pretrained weights may be partial
            # so relax the strict condition.
            model.load_state_dict(weights, strict=(self.mode == 'resume'))

        # Move to cuda
        self.model.cuda()

        # Print model topology and number of parameters
        self.print(self.model)
        self.print("# parameters: {} ({} learnable)".format(
            *get_n_params(self.model)))

        # Reseed to retain the order of shuffle operations
        if self.seed != 0:
            np.random.seed(self.seed)

        # Create optimizer instance
        self.optim = Optimizer(
            self.optimizer, self.model, lr=self.lr,
            weight_decay=self.l2_reg, gclip=self.gclip)
        self.print(self.optim)

        # Create TensorBoard logger if possible and requested
        self.tb = TensorBoard(self.model, self.tensorboard_dir,
                              self.exp_id, self.subfolder)
        self.print(self.tb)

    def train_epoch(self):
        """Trains a full epoch."""
        self.print('Starting Epoch {}'.format(self.monitor.ectr))

        total_loss = 0.0
        n_tokens_seen = 0

        nn_sec = 0.0
        eval_sec = 0.0
        total_sec = time.time()

        for batch in self.train_iterator:
            #############################
            self.monitor.uctr += 1
            nn_start = time.time()

            # Reset gradients
            self.optim.zero_grad()

            # Forward pass returns mean data loss
            loss = self.model(to_var(batch))
            batch_loss = loss.data.cpu()[0]

            # Get auxiliary loss if any
            aux_loss = 0
            if self.model.aux_loss is not None:
                loss += self.model.aux_loss
                aux_loss = self.model.aux_loss.data.cpu()[0]

            # Backward pass
            loss.backward()

            # Update parameters (includes gradient clipping logic)
            self.optim.step()

            # Keep stats
            nn_sec += (time.time() - nn_start)
            #############################

            total_loss += batch_loss * self.model.n_tokens
            n_tokens_seen += self.model.n_tokens

            if self.monitor.uctr % self.disp_freq == 0:
                self.print("Epoch {} - update {:10d} => loss: {:.3f} "
                           "(aux_loss: {:.4f})".format(self.monitor.ectr,
                                                       self.monitor.uctr,
                                                       batch_loss,
                                                       aux_loss))

                # Send statistics
                self.tb.log_scalar('train_LOSS', batch_loss, self.monitor.uctr)

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

        # Compute epoch loss
        epoch_loss = total_loss / n_tokens_seen
        self.monitor.train_loss.append(epoch_loss)

        # All time spent for this epoch
        total_min = (time.time() - total_sec) / 60
        # All time spent during forward/backward/step
        nn_min = nn_sec / 60
        # All time spent during validation(s)
        eval_min = eval_sec / 60
        # Rest is iteration overhead + checkpoint saving
        overhead_min = total_min - nn_min - eval_min

        self.print("--> Epoch {} finished with mean loss {:.5f}".format(
            self.monitor.ectr, epoch_loss))
        self.print("--> Overhead/Training/Evaluation: {:.2f}/{:.2f}/{:.2f} "
                   "mins (total: {:.2f} mins)   ({} sent/sec)".format(
                       overhead_min, nn_min, eval_min, total_min,
                       int(len(self.model.datasets['train']) / nn_sec)))

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

        if 'LOSS' in self.monitor.eval_metrics:
            # Compute validation loss and add it
            val_loss = self.model.compute_loss(self.vloss_iterator)
            results.append(Metric('LOSS', val_loss, higher_better=False))

        if self.monitor.beam_metrics:
            self.print('Performing translation search (beam_size:{})'.format(
                self.eval_beam))
            beam_time = time.time()
            hyps = beam_search(self.model, self.beam_iterator,
                               self.model.trg_vocab, beam_size=self.eval_beam)
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

        self.print('Training finished on %s' % time.strftime('%d-%m-%Y %H:%M'))
        # Close tensorboard
        self.tb.close()
