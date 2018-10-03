import os
import shutil
import subprocess

import torch

DEVICE = None


class DeviceManager:
    __errors = {
        'NoDevFiles': 'Make sure you requested a GPU resource from your cluster.',
        'NoSMI': 'nvidia-smi is not installed. Are you on the correct node?',
        'EnvVar': 'Please set CUDA_VISIBLE_DEVICES explicitly.',
        'NotEnoughGPU': 'You requested more GPUs than it is available.',
        'NoMultiGPU': 'Multi-GPU not supported, please restrict CUDA_VISIBLE_DEVICES',
    }

    def __init__(self, dev='cuda'):
        # What user requests
        self.dev = dev
        self.req_cpu = dev == 'cpu'
        self.req_gpu = dev.startswith('cuda')
        self.req_multi_gpu = self.req_gpu and ',' in dev
        self.req_n_gpu = len(dev[5:].split(',')) if self.req_gpu else 0
        self.pid = os.getpid()

        # What we have
        self.nvidia_smi = shutil.which('nvidia-smi')
        self.cuda_vis_dev = os.environ.get('CUDA_VISIBLE_DEVICES', None)

        if self.req_gpu:
            self.__check_gpu_setup()

        global DEVICE
        DEVICE = torch.device(dev)

    def __check_gpu_setup(self):
        if self.nvidia_smi is None:
            raise RuntimeError(self.__errors['NoSMI'])
        if self.cuda_vis_dev == "NoDevFiles":
            raise RuntimeError(self.__errors['NoDevFiles'])
        elif self.cuda_vis_dev is None:
            raise RuntimeError(self.__errors['EnvVar'])

        # How many GPUs do we have access to?
        self.avail_gpus = self.cuda_vis_dev.split(',')
        self.n_avail_gpus = len(self.avail_gpus)

        # NOTE: Remove this once multi-GPU is supported
        if self.req_n_gpu > 1 or self.n_avail_gpus > 1:
            raise RuntimeError(self.__errors['NoMultiGPU'])

        if self.req_n_gpu > self.n_avail_gpus:
            raise RuntimeError(self.__errors['NotEnoughGPU'])

    def get_cuda_mem_usage(self, name=True):
        if self.cpu:
            return None

        p = subprocess.run([
            self.nvidia_smi,
            "--query-compute-apps=pid,gpu_name,used_memory",
            "--format=csv,noheader"], stdout=subprocess.PIPE, universal_newlines=True)

        for line in p.stdout.strip().split('\n'):
            pid, gpu_name, usage = line.split(',')
            if int(pid) == self.pid:
                if name:
                    return '{} -> {}'.format(gpu_name.strip(), usage.strip())
                else:
                    return usage.strip()

        return 'N/A'

    def get_free_gpus(self):
        """Deprecated: please explicitly give CUDA_VISIBLE_DEVICES."""
        if self.cpu:
            return None

        p = subprocess.run([self.nvidia_smi, "-q", "-d", "PIDS"],
                           stdout=subprocess.PIPE,
                           universal_newlines=True)

        # Boolean map
        self.free_map = ["None" in l for l in p.stdout.split('\n')
                         if "Processes" in l]

        self.free_count = self.free_map.count(True)
        self.free_idxs = [i for i in range(len(self.free_map))
                          if self.free_map[i]]

    def __repr__(self):
        if self.req_cpu:
            return "DeviceManager(dev='cpu')"
        else:
            return "DeviceManager(dev='cuda', n_gpu={})".format(self.req_n_gpu)
