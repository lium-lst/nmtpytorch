# -*- coding: utf-8 -*-
import pathlib

from torch.utils.tensorboard import SummaryWriter


class TensorBoard:
    def __init__(self, model, log_dir, exp_id, subfolder):
        self.model = model
        self.log_dir = log_dir
        self.exp_id = exp_id
        self.subfolder = subfolder
        self.writer = None
        self.available = bool(self.log_dir)

        # Call setup
        self.setup()

    def _nop(self, *args, **kwargs):
        return

    def setup(self):
        """Setups TensorBoard logger."""
        if not self.available:
            self.replace_loggers()
            return

        # Construct full folder path
        self.log_dir = pathlib.Path(self.log_dir).expanduser()
        self.log_dir = self.log_dir / self.subfolder / self.exp_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up summary writer
        self.writer = SummaryWriter(self.log_dir)

    def replace_loggers(self):
        """Replace all log_* methods with dummy _nop."""
        self.log_metrics = self._nop
        self.log_scalar = self._nop
        self.log_activations = self._nop
        self.log_gradients = self._nop

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

    def close(self):
        """Closes TensorBoard handle."""
        if self.available:
            self.writer.close()

    def __repr__(self):
        if not self.log_dir:
            return "No 'tensorboard_dir' given in config"
        return "TensorBoard is active"
