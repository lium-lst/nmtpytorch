from functools import lru_cache
from pathlib import Path

from PIL import Image

import torch
from torchvision import transforms

from .base import BaseDataset


class ImageFolderDataset(BaseDataset):
    """A variant of torchvision.datasets.ImageFolder which drops support for
    target loading, i.e. this only loads images not attached to any other
    label.

    This class also makes use of ``lru_cache`` to cache an image file once
    opened to avoid repetitive disk access.

    Arguments:
        root (str): The root folder that contains the images and index.txt
        resize (int, optional): An optional integer to be given to
            ``torchvision.transforms.Resize``. Default: ``None``.
        crop (int, optional): An optional integer to be given to
            ``torchvision.transforms.CenterCrop``. Default: ``None``.
        replicate(int, optional): Replicate the image names ``replicate``
            times in order to process the same image ``replicate`` times
            if ``replicate`` sentences are available during training time.
        warmup(bool, optional): If ``True``, the images will be read once
            at the beginning to fill the cache.

    """
    def __init__(self, root, resize=None, crop=None,
                 replicate=1, warmup=False):
        self.root = Path(root).expanduser().resolve()
        self.replicate = replicate
        self.warmup = warmup
        self._image_files = []

        # Image list in dataset order
        self.index = self.root / 'index.txt'
        assert self.index.exists(), f"{self.index} could not be found"

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

        with self.index.open() as f:
            for fname in f:
                fname = self.root / fname.strip()
                assert fname.exists(), "{} does not exist.".format(fname)
                self._image_files.append(str(fname))

        # Setup reader
        self.read_image = lru_cache(maxsize=self.__len__())(self._read_image)

        if self.warmup:
            for idx in range(self.__len__()):
                self[idx]

        # Replicate the list if requested
        self._image_files = self._image_files * self.replicate

    def _read_image(self, fname):
        with open(fname, 'rb') as f:
            img = Image.open(f).convert('RGB')
            return self.transform(img)

    def collate(elems):
        return torch.stack(elems)

    def __getitem__(self, idx):
        return self.read_image(self._image_files[idx])

    def __len__(self):
        return len(self._image_files)
