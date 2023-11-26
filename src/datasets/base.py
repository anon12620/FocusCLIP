from abc import ABC, abstractmethod

from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        self._root = root
        self._split = split
        self._transform = transform
        self._target_transform = target_transform
        self._samples = self._load_split_data(split)

    @abstractmethod
    def _load_split_data(self, split) -> list:
        """ Load the data for the specified split.

        This method should be implemented by subclasses to perform the
        actual loading of data. The returned list should contain tuples
        of (image_path, target) pairs, where image_path is a string
        containing the path to the image and target is a dictionary
        containing the target annotations for the image.

        Args:
            split: The split to load data for.

        Returns:
            A list of (image_path, target) tuples.
        """
        pass

    def _load_sample(self, idx):
        """ Load an untransformed sample from the dataset. """
        image_path, target = self._samples[idx]
        image = Image.open(image_path).convert('RGB')
        return image, target

    def _transform_image(self, image, target):
        """ Apply the image transform, if specified. """
        return self._transform(image) if self._transform else image

    def _transform_target(self, target):
        """ Apply the target transform, if specified. """
        return self._target_transform(target) if self._target_transform else target

    def __getitem__(self, idx):
        """ Load and transform a sample from the dataset. """
        # Load the sample
        image, target = self._load_sample(idx)

        # Apply transformations
        image = self._transform_image(image, target)
        target = self._transform_target(target)

        return image, target

    def __len__(self):
        return len(self._samples)