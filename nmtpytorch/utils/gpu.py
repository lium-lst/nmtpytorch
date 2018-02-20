import os
import subprocess


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

        self.free_count = self.free_map.count(True)
        self.free_idxs = [i for i in range(len(self.free_map))
                          if self.free_map[i]]

    def __call__(self, devs, strict=False):
        vis_dev = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if vis_dev is not None:
            if vis_dev == "NoDevFiles":
                raise RuntimeError("No GPU found.")
            else:
                return vis_dev

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

        if strict:
            _devices = ",".join([str(dev) for dev in devices])
            os.environ['CUDA_VISIBLE_DEVICES'] = _devices

        # Return list of indices
        return devices
