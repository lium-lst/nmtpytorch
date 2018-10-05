# -*- coding: utf-8 -*-

import pathlib


class TensorBoard:
    def __init__(self, model, log_dir, exp_id, subfolder,
                 send_metrics=True, send_activations=False,
                 send_gradients=False):

        self.model = model
        self.log_dir = log_dir
        self.exp_id = exp_id
        self.subfolder = subfolder
        self.send_metrics = send_metrics
        self.send_activations = send_activations
        self.send_gradients = send_gradients
        self.writer = None
        self.available = False

        # Call setup
        self.setup()

    def __repr__(self):
        if self.available:
            return "TensorBoard is active"
        else:
            if not self.log_dir:
                return "No 'tensorboard_dir' given in config"
            else:
                return "TensorboardX not installed"

    def setup(self):
        """Setups TensorBoard logger."""

        def replace_loggers():
            # Replace all log_* methods with dummy _nop
            self.log_metrics = self._nop
            self.log_scalar = self._nop
            self.log_activations = self._nop
            self.log_gradients = self._nop

        # No log_dir given, bail out
        if not self.log_dir:
            replace_loggers()
            return

        # Detect tensorboard
        try:
            from tensorboardX import SummaryWriter
        except ImportError as ie:
            replace_loggers()
            return
        else:
            self.available = True

            # Construct full folder path
            self.log_dir = pathlib.Path(self.log_dir).expanduser()
            self.log_dir = self.log_dir / self.subfolder / self.exp_id
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # Set up summary writer
            self.writer = SummaryWriter(self.log_dir)

    def _nop(self, *args, **kwargs):
        return

    def close(self):
        """Closes TensorBoard handle."""
        if self.available:
            self.writer.close()

    def log_metrics(self, metrics, step, suffix=''):
        """Logs evaluation metrics as scalars."""
        for metric in metrics:
            self.writer.add_scalar(suffix + metric.name, metric.score,
                                   global_step=step)

    def log_scalar(self, name, value, step):
        """Logs single scalar value."""
        self.writer.add_scalar(name, value, global_step=step)

    def log_activations(self, step):
        """Logs activations by layer."""
        pass

    def log_gradients(self, step):
        """Logs gradients by layer."""
        pass
