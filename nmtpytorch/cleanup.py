# -*- coding: utf-8 -*-
import os
import sys
import signal
import atexit
import pathlib
import traceback


class Cleanup:
    def __init__(self):
        self.temp_files = set()
        self.processes = set()

    def register_tmp_file(self, tmp_file):
        """Add new temp file to global set."""
        self.temp_files.add(pathlib.Path(tmp_file))

    def register_proc(self, pid):
        """Add new process to global set."""
        self.processes.add(pid)

    def unregister_proc(self, pid):
        """Remove given PID from global set."""
        self.processes.remove(pid)

    def __call__(self):
        """Cleanup registered temp files and kill PIDs."""
        for tmp_file in filter(lambda x: x.exists(), self.temp_files):
            tmp_file.unlink()

        for proc in self.processes:
            try:
                os.kill(proc, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def __repr__(self):
        repr_ = "Cleanup Manager\n"
        if len(self.processes) > 0:
            repr_ += "Tracking Processes\n"
            for proc in self.processes:
                repr_ += " {}\n".format(proc)

        if len(self.temp_files) > 0:
            repr_ += "Tracking Temporary Files\n"
            for tmp_file in self.temp_files:
                repr_ += " {}\n".format(tmp_file)

        return repr_

    @staticmethod
    def register_exception_handler(logger, quit_on_exception=False):
        """Setup exception handler."""

        def exception_handler(exctype, val, trace):
            """Let Python call this when an exception is uncaught."""
            logger.info(
                ''.join(traceback.format_exception(exctype, val, trace)))

        def exception_handler_quits(exctype, val, trace):
            """Let Python call this when an exception is uncaught."""
            logger.info(
                ''.join(traceback.format_exception(exctype, val, trace)))
            sys.exit(1)

        if quit_on_exception:
            sys.excepthook = exception_handler_quits
        else:
            sys.excepthook = exception_handler

    @staticmethod
    def register_handler(logger, _atexit=True, _signals=True,
                         exception_quits=False):
        """Register atexit and signal handlers."""
        if _atexit:
            # Register exit handler
            atexit.register(cleanup)

        if _signals:
            # Register SIGINT and SIGTERM
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

        Cleanup.register_exception_handler(logger, exception_quits)


# Create a global cleaner
cleanup = Cleanup()


def signal_handler(signum, frame):
    """Let Python call this when SIGINT or SIGTERM caught."""
    cleanup()
    sys.exit(0)
