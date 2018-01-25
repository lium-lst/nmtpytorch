import os
import sys
import time
import pathlib
import subprocess

from .misc import get_temp_file, listify


class GPUManager(object):

    def collect_stats(self):
        try:
            p = subprocess.run(["nvidia-smi", "-q", "-d", "PIDS"],
                               stdout=subprocess.PIPE,
                               universal_newlines=True)
        except FileNotFoundError as oe:
            raise FileNotFoundError("nvidia-smi not found!")

        # Boolean map
        self.free_map = ["None" in l for l in p.stdout.split('\n')
                         if "Processes" in l]
        for i in range(len(self.free_map)):
            if self.free_map[i]:
                lock_path = pathlib.Path('/tmp/nmtpy.gpu%d.lock' % i)
                if lock_path.exists():
                    lock_str = lock_path.read_text().strip()
                    pid = lock_str.split(':')[-1]
                    if not (pathlib.Path('/proc') / pid).exists():
                        lock_path.unlink()
                    else:
                        self.free_map[i] = False

        self.free_count = self.free_map.count(True)
        self.free_idxs = [i for i in range(len(self.free_map))
                          if self.free_map[i]]

        self.pid = os.getpid()

    def lock(self, devs):
        for dev in listify(devs):
            name = "nmtpy.gpu%d.lock" % (dev)
            if self.free_map[dev]:
                lockfile = get_temp_file(name=name)
                lockfile.write("%s pid:%d\n" % (name, self.pid))
                self.free_map[dev] = False

    def __call__(self, devs, strict=False):
        self.collect_stats()

        if not isinstance(devs, str):
            devs = str(devs)

        if devs.startswith('auto_'):
            # Find out how many GPUs requested
            how_many = int(devs.split('_')[-1])

            if how_many > self.free_count:
                raise Exception("Less than {} GPUs available.".format(
                    how_many))
            devices = self.free_idxs[:how_many]
        else:
            # Manual list of GPU(s) given
            devices = [int(d) for d in devs.split(',')]
            if not all(self.free_map[i] for i in devices):
                raise Exception("Not all requested GPUs are available.")

        # Finally lock devices
        self.lock(devices)

        if strict:
            _devices = ",".join([str(dev) for dev in devices])
            os.environ['CUDA_VISIBLE_DEVICES'] = _devices

        # Return list of indices
        return devices
