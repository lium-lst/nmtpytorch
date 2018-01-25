# -*- coding: utf-8 -*-
from functools import lru_cache
from pathlib import Path

from PIL import Image

import torch.utils.data as data
from torchvision import transforms


class ImageFolderDataset(data.Dataset):
    """A variant of torchvision.datasets.ImageFolder which drops support for
    target loading, i.e. this only loads images not attached to any other
    label.

    This class also makes use of ``lru_cache`` to cache an image file once
    opened to avoid repetitive disk access.

    Arguments:
        root (str): The root folder which contains a folder per each split.
        split (str): A subfolder that should exist under ``root`` containing
            images for a specific split.
        resize (int, optional): An optional integer to be given to
            ``torchvision.transforms.Resize``. Default: ``None``.
        crop (int, optional): An optional integer to be given to
            ``torchvision.transforms.CenterCrop``. Default: ``None``.

    """
    def __init__(self, root, split, resize=None, crop=None):
        self.split = split
        self.root = Path(root).expanduser().resolve() / self.split
        # Image list in dataset order
        self.index = self.root / 'index.txt'

        _transforms = []
        if resize is not None:
            _transforms.append(transforms.Resize(resize))
        if crop is not None:
            _transforms.append(transforms.CenterCrop(crop))
        _transforms.append(transforms.ToTensor())
        _transforms.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))
        self.transform = transforms.Compose(_transforms)

        if not self.index.exists():
            raise(RuntimeError(
                "index.txt does not exist in {}".format(self.root)))

        self.image_files = []
        with self.index.open() as f:
            for fname in f:
                fname = self.root / fname.strip()
                assert fname.exists(), "{} does not exist.".format(fname)
                self.image_files.append(str(fname))

        # Setup reader
        self.read_image = lru_cache(maxsize=self.__len__())(self._read_image)

    def _read_image(self, fname):
        with open(fname, 'rb') as f:
            img = Image.open(f).convert('RGB')
            return self.transform(img)

    def __getitem__(self, idx):
        return self.read_image(self.image_files[idx])

    def __len__(self):
        return len(self.image_files)

    def __repr__(self):
        s = "{}({})\n".format(self.__class__.__name__, self.root)
        s += " split '{}' - {} images\n".format(self.split, self.__len__())
        if self.transform:
            s += ' Transforms: {}\n'.format(
                self.transform.__repr__().replace('\n', '\n' + ' '))
        return s
