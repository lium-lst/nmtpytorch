# -*- coding: utf-8 -*-
import time
import math
import pathlib
from collections import OrderedDict

import numpy as np

import torch

from nmtpytorch.metrics import Evaluator
from nmtpytorch.utils.misc import force_symlink, get_n_params
from nmtpytorch.utils.data import to_var
from nmtpytorch.utils.tensorboard import TensorBoard
from nmtpytorch.optimizer import Optimizer


class MainLoop(object):
    def __init__(self, model, logger, train_opts, history, model_weights,
                 mode='train'):
        # Set print function to logger's info()
        self.print = logger.info

        # Set model
        self.model = model

        # Get all training options into this mainloop
        self.__dict__.update(train_opts)

        # Convert save_path to pathlib for easy manipulation
        self.save_path = pathlib.Path(self.save_path)

        # Initialize stateful variables from zero or checkpoint
        self.load_state(history)

        self.print('Loading dataset(s)')
        self.model.load_data('train')
        self.model.load_data('val')
        self.train_iterator = self.model.datasets['train'].get_iterator(
            self.batch_size)
        self.valloss_iterator = self.model.datasets['val'].get_iterator(
            self.batch_size)
        # For batched GPU beam-search, an effective batch_size of 48
        # seems to give good speed
        self.beam_iterator = self.model.datasets['val'].get_iterator(
            48 // self.eval_beam, only_source=True)

        # If pretrained weights are given, evaluate before starting
        self.immediate_eval = (mode == 'train') and model_weights

        # Setup model
        self.model.setup()

        # This should come after model.setup()
        if model_weights is not None:
            self.print('Loading previous model weights')
            # If not resuming -> pretrained weights are loaded from an
            # arbitrary model, relax the strict condition.
            model.load_state_dict(model_weights, strict=(mode == 'resume'))

        # Move to cuda
        self.model.cuda()

        # Print number of parameters
        self.print("# parameters: {} ({} learnable)".format(
            *get_n_params(self.model)))

        # Reseed to retain the order of shuffle operations
        if self.seed != 0:
            np.random.seed(self.seed)

        # Add an attribute for end-of-epoch validations
        self.epoch_valid = (self.eval_freq == 0)

        # Create optimizer instance
        self.optim = Optimizer(
            self.optimizer, self.model,
            lr=self.lr, weight_decay=self.l2_reg, gclip=self.gclip)

        self.print(self.optim)

        # We may have no validation data.
        self.beam_metrics = []
        if self.eval_freq >= 0:

            # Requested metrics
            metrics = self.eval_metrics.split(',')

            # first one is for early-stopping
            self.early_metric = metrics[0]

            for metric in metrics:
                if metric not in self.evals:
                    self.evals[metric] = []

            # Prepare the string to pass to beam_search
            self.beam_metrics = [m for m in self.evals if m != 'loss']

        # Create evaluator object
        self.evaluator = Evaluator(self.model.val_refs, self.beam_metrics,
                                   filters=self.eval_filters)

        # Create TensorBoard logger if possible and requested
        self.tb = TensorBoard(self.model, self.tensorboard_dir, self.exp_id,
                              self.subfolder)
        # Print information
        self.print(self.tb)

    def save_best_model(self):
        """Saves best N models to disk."""
        # Get the score of the system that will be saved
        cur_score = self.evals[self.early_metric][-1]

        # Custom filename with metric score
        cur_fname = "%s-val%3.3d-%s_%.3f.pt" % (self.exp_id, self.vctr,
                                                self.early_metric,
                                                cur_score)

        # Stack is empty, save the model whatsoever
        if len(self.best_models) < self.save_best_n:
            self.best_models.append((cur_score, cur_fname))

        # Stack is full, replace the worst model
        else:
            prune = self.best_models[self.next_prune_idx][1]
            (self.save_path / self.subfolder / prune).unlink()
            self.best_models[self.next_prune_idx] = (cur_score, cur_fname)

        self.print('Saving model with best validation %s' %
                   self.early_metric.upper())
        self.save(cur_fname)

        # Create a .best symlink
        symlink = str(self.save_path / self.subfolder / self.exp_id) + '.best'
        force_symlink(cur_fname, symlink, relative=True)

        # In the next best, we'll remove the following idx from the store
        # Metric specific comparator stuff
        where = self.evaluator.comparators[self.early_metric][-1]
        getitem = self.best_models.__getitem__
        self.next_prune_idx = sorted(range(len(self.best_models)),
                                     key=getitem)[where]

    def train_epoch(self):
        """Train a full epoch."""

        # Even if we resume, we'll start over a clean epoch
        self.print('Starting Epoch %d' % self.ectr)
        epoch_time = 0.0
        start_uctr = self.uctr
        n_tokens_seen = 0
        total_loss = 0.0

        # Iterate over batches
        for batch in self.train_iterator:
            start_time = time.time()

            # Increment update counter
            self.uctr += 1

            # Reset gradients
            self.optim.zero_grad()

            # Forward pass returns mean data loss
            loss = self.model(to_var(batch))

            # Get other losses
            aux_loss = self.model.aux_loss()

            # Backward pass
            (loss + aux_loss).backward()

            # Update parameters (includes gradient clipping logic)
            self.optim.step()

            # Accumulate time spent
            epoch_time += (time.time() - start_time)

            # Keep stats
            batch_loss = loss.data.cpu()[0]
            total_loss += batch_loss * self.model.n_tokens
            n_tokens_seen += self.model.n_tokens

            # Verbose
            if self.uctr % self.disp_freq == 0:
                self.print("Epoch: %6d, update: %7d, loss: %5.3f "
                           "(aux_loss: %5.3f)" %
                           (self.ectr, self.uctr, batch_loss, aux_loss))

                # Send statistics
                self.tb.log_scalar('train_loss', batch_loss, self.uctr)

            # Checkpoint?
            if self.checkpoint_freq and self.uctr % self.checkpoint_freq == 0:
                self.save(self.exp_id, checkpoint=True)

            # Should we stop
            if self.uctr == self.max_iterations:
                self.print("Max iterations %d reached." % self.uctr)
                return False

            # Do validation
            if not self.epoch_valid and self.ectr >= self.eval_start \
                    and self.eval_freq > 0 \
                    and self.uctr % self.eval_freq == 0:
                self.do_validation()

            # Check stopping conditions
            if self.early_bad == self.patience:
                self.print("Early stopped.")
                return False

        # Compute epoch loss
        epoch_loss = total_loss / n_tokens_seen
        self.epoch_losses.append(epoch_loss)

        self.print(
            "--> Epoch %d finished in %.3f minutes (%.3f sec/batch)"
            " with mean loss %.5f (PPL: %4.5f)" %
            (self.ectr, epoch_time / 60,
             epoch_time / (self.uctr - start_uctr),
             epoch_loss, np.exp(epoch_loss)))

        # Do validation?
        if self.epoch_valid and self.ectr >= self.eval_start:
            self.do_validation()

        # Check whether maximum epoch is reached
        if self.ectr == self.max_epochs:
            self.print("Max epochs %d reached." % self.max_epochs)
            return False

        # Increment now for the next epoch
        self.ectr += 1

        return True

    def do_validation(self):
        """Do early-stopping validation."""
        self.model.train(False)
        self.vctr += 1

        # Compute validation loss
        cur_loss = self.model.compute_loss(self.valloss_iterator)

        # Add val_loss
        self.evals['loss'].append(cur_loss)

        # Print validation loss
        self.print("Validation %2d - LOSS = %.3f (PPL: %.3f)" %
                   (self.vctr, cur_loss, math.exp(cur_loss)))

        self.tb.log_scalar('val_loss', cur_loss, self.uctr)

        if any(self.beam_metrics):
            self.print('Performing translation search (beam_size:{})'.format(
                self.eval_beam))
            beam_time = time.time()
            hyps = self.model.beam_search(self.beam_iterator, k=self.eval_beam)
            spent = (time.time() - beam_time) / 60.
            self.print('Completed in %.3f minutes.' % spent)

            # Compute metrics
            results = list(self.evaluator.score(hyps).values())

            # Send to tensorboard
            self.tb.log_metrics(results, self.uctr)

            # Metric() instances are printable
            for metric in results:
                self.print("Validation %2d - %s" % (self.vctr, metric))
                self.evals[metric.name.lower()].append(metric.score)

        # Check whether this is the best or not
        if self.evaluator.is_last_best(self.early_metric,
                                       self.evals[self.early_metric],
                                       self.patience_delta):
            self.early_bad = 0
            if self.save_best_n > 0:
                self.save_best_model()
        else:
            self.early_bad += 1
            self.print("Early stopping patience: %d left" %
                       (self.patience - self.early_bad))

        # Dump summary
        self.dump_val_summary()
        self.model.train(True)

    def dump_val_summary(self):
        """Print validation summary."""
        for metric, history in self.evals.items():
            # Find the best validation idx and value so far
            best_idx, best_val = self.evaluator.find_best(metric, history)
            if metric == 'loss':
                msg = "Best %s = %.2f (PPL: %.2f)" % (metric.upper(),
                                                      best_val,
                                                      np.exp(best_val))
            else:
                msg = "Best %s = %.2f" % (metric.upper(), best_val)

            self.print('--> Current %s at validation %d' % (msg, best_idx))

        # Remember who we are
        self.print('--> This is model: %s' % self.exp_id)

    def run(self):
        """Runs training loop."""
        if self.immediate_eval:
            self.do_validation()

        self.model.train(True)
        self.print('Training started on %s' % time.strftime('%d-%m-%Y %H:%M'))
        while self.train_epoch():
            pass

        # Final summary
        if len(self.evals['loss']) > 0:
            self.dump_val_summary()
        else:
            # No validation data used, save the final model
            self.print('Saving final model.')
            self.save(self.exp_id)

        self.print('Training finished on %s' % time.strftime('%d-%m-%Y %H:%M'))
        # Close tensorboard
        self.tb.close()

    def save(self, fname, checkpoint=False):
        """Saves the current model optionally with stateful variables."""
        d = {'opts': self.model.opts.to_dict(),
             'model': self.model.state_dict()}
        if checkpoint:
            d['history'] = self.get_state()
            fname = fname if fname.endswith('.ckpt') else fname + '.ckpt'
        else:
            fname = fname if fname.endswith('.pt') else fname + '.pt'

        torch.save(d, self.save_path / self.subfolder / fname)

    def load_state(self, history):
        """Restores stateful variables from dictionary if not empty."""
        if history:
            self.print('Restoring previous history.')

        states = {'uctr': history.pop('uctr', 0),   # update ctr
                  'ectr': history.pop('ectr', 1),   # epoch ctr
                  'vctr': history.pop('vctr', 0),   # validation ctr
                  'early_bad': history.pop('early_bad', 0),
                  'epoch_losses': history.pop('epoch_losses', []),
                  'best_models': history.pop('save_best_n', []),
                  'evals': history.pop('evals', OrderedDict({'loss': []}))}
        self._stateful = list(states.keys())
        self.__dict__.update(states)

    def get_state(self):
        """Returns the current values of stateful variables."""
        return {k: getattr(self, k) for k in self._stateful}
