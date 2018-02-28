# -*- coding: utf-8 -*-
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import DataLoader

from . import ImageFolderDataset, TextDataset, OneHotDataset
from . import MultiParallelDataset
from .collate import get_collate_v2

from ..samplers import BucketBatchSampler


class Multi30kRawDataset(object):
    """Returns a Dataset for Multi30k using raw JPG images.

    Arguments:
        data_dict(dict): [data] section's relevant split dictionary
        vocabs(dict): dictionary mapping lang keys to Vocabulary() objects
        topology(Topology): A topology object.
        warmup(bool, optional): If ``True``, raw images will be processed
            and cached once.
        resize (int, optional): An optional integer to be given to
            ``torchvision.transforms.Resize``. Default: ``256``.
        crop (int, optional): An optional integer to be given to
            ``torchvision.transforms.CenterCrop``. Default: ``224``.
        replicate(int, optional): Replicate the images ``replicate``
            times in order to process the same image that many times
            if ``replicate`` sentences are available during training time.
    """
    def __init__(self, data_dict, vocabs, topology,
                 warmup=False, resize=256, crop=224, replicate=1):

        self.topology = topology
        data = {}

        for src, ds in self.topology.srcs.items():
            # Remove from topology if no data provided (possible in test time)
            if src not in data_dict:
                del self.topology.srcs[src]
                continue
            if ds._type.startswith(("Text", "OneHot")):
                Dataset = globals()['{}Dataset'.format(ds._type)]
                data[src] = Dataset(data_dict[src], vocabs[src])
            elif ds._type == "ImageFolder":
                data[src] = ImageFolderDataset(
                    data_dict[src], resize=resize,
                    crop=crop, replicate=replicate, warmup=warmup)

        for trg, ds in self.topology.trgs.items():
            # Remove from topology if no data provided (possible in test time)
            if trg not in data_dict:
                del self.topology.trgs[trg]
                continue
            path = data_dict[trg]
            if ds._type == "Text":
                data[trg] = TextDataset(path, vocabs[trg], bos=True)
            elif ds._type == "OneHot":
                data[trg] = OneHotDataset(path, vocabs[trg])

        # The keys (DataSource()) convey information about data sources
        self.dataset = MultiParallelDataset(
            src_datasets={v: data[k] for k, v in self.topology.srcs.items()},
            trg_datasets={v: data[k] for k, v in self.topology.trgs.items()},
        )

    def get_iterator(self, batch_size, only_source=False):
        if only_source:
            # Sequential batches, no shuffling, no bucketing
            sampler = SequentialSampler(self.dataset)
            return DataLoader(
                self.dataset, sampler=sampler,
                shuffle=False, batch_size=batch_size,
                collate_fn=get_collate_v2(self.dataset.sources))
        else:
            # Target-length bucketed, shuffled batches
            target_lengths = self.dataset.data[self.dataset.targets[0]].lengths

            sampler = BucketBatchSampler(target_lengths, batch_size)
            return DataLoader(
                self.dataset, batch_sampler=sampler,
                collate_fn=get_collate_v2(self.dataset.data_sources))

    def __repr__(self):
        return self.dataset.__repr__()

    def __len__(self):
        return len(self.dataset)
