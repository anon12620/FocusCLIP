from abc import ABC, abstractmethod

from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """ Base class for all datasets. """

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
        image, target = self._load_sample(idx)
        image = self._transform_image(image, target)
        target = self._transform_target(target)
        return image, target

    def __len__(self):
        return len(self._samples)


class AgeGenderDataset(BaseDataset):
    """ Base class for all datasets containing age and gender annotations. """

    def __init__(self, root, split='train', **kwargs):
        super().__init__(root, split, **kwargs)
        assert hasattr(self, 'age_groups'), 'self.age_groups must be defined in _load_split_data'
        self.num_age_groups = len(self.age_groups)  # type: ignore
        self.captions_age = [self._map_age(a) for a in self.age_groups]  # type: ignore

        assert hasattr(self, 'genders'), 'self.genders must be defined in _load_split_data'
        self.num_genders = len(self.genders)  # type: ignore
        self.captions_gender = [self._map_gender(g) for g in self.genders]  # type: ignore

    @property
    def captions(self):
        _captions = []
        for a in self.age_groups:  # type: ignore
            for g in self.genders:  # type: ignore
                _captions.append(self._map_age_and_gender(a, g))
        return _captions

    def _map_age(self, age_group):
        assert age_group in self.age_groups, f'age {age_group} not in {self.age_groups}'  # type: ignore
        return f'a photo of a {age_group.lower()} person'

    def _map_gender(self, gender):
        assert gender in self.genders, f'gender {gender} not in {self.genders}'  # type: ignore
        return f'a photo of a {gender.lower()} person'

    def _map_age_and_gender(self, age_group, gender):
        assert age_group in self.age_groups, f'age {age_group} not in {self.age_groups}'  # type: ignore
        assert gender in self.genders, f'gender {gender} not in {self.genders}'  # type: ignore
        return f'a photo of a {gender.lower()} {age_group.lower()} person'


class AgeGenderRaceDataset(AgeGenderDataset):
    """ Base class for all datasets containing age, gender and race annotations. """

    def __init__(self, root, split='train', **kwargs):
        super().__init__(root, split, **kwargs)
        assert hasattr(self, 'races'), 'self.races is not defined in the subclass'
        self.num_races = len(self.races)  # type: ignore
        self.captions_race = [self._map_race(r) for r in self.races]  # type: ignore

    def _map_race(self, race):
        assert race in self.races, f'race {race} not in {self.races}'  # type: ignore
        if race == 'others':
            race = 'other race'
        if race[0] in ['a', 'e', 'i', 'o', 'u']:
            article = 'an'
        else:
            article = 'a'
        return f'a photo of {article} {race.lower()} person'
