import os

import pandas as pd

from ._base import AgeGenderRaceDataset


class UTKFaceDataset(AgeGenderRaceDataset):
    """ The UTKFace dataset. """

    def __init__(self, root='data/eval/utkface', split='train', **kwargs):
        assert split in ['train', 'val']
        super().__init__(root, split, **kwargs)

    def _load_split_data(self, split):
        # Read the annotations
        anno_file = os.path.join(self._root, 'annotations', f'{split}.csv')
        df = pd.read_csv(anno_file)

        self.age_groups = ['baby', 'child', 'teenager', 'young adult', 'adult']
        self.races = ['white', 'black', 'asian', 'indian', 'others']
        self.genders = ['male', 'female']

        # Filter out invalid images
        images = [n.split('/')[-1] for n in df['img_name']]
        images_dir = os.path.join(self._root, 'images')
        images = [i + '.chip.jpg' for i in images
                  if os.path.exists(os.path.join(images_dir, i + '.chip.jpg'))]

        # Create the samples
        samples = []
        for image_name in images:
            age = int(image_name.split('_')[0])
            gender_id = int(image_name.split('_')[1])
            race_id = int(image_name.split('_')[2])

            age_group = self.get_age_category(age)

            image_path = os.path.join(images_dir, image_name)
            samples.append((image_path, {
                'age': age,
                'age_group': age_group,
                'age_group_id': self.age_groups.index(age_group),
                'gender': self.genders[gender_id],
                'gender_id': gender_id,
                'race': self.races[race_id],
                'race_id': race_id,
            }))
        return samples

    def get_age_category(self, age):
        age = int(age)
        if age <= 2:
            return self.age_groups[0]  # 0-2 is baby
        elif age <= 9:
            return self.age_groups[1]  # 3-9 is child
        elif age <= 19:
            return self.age_groups[2]  # 10-19 is teenager
        elif age <= 29:
            return self.age_groups[3]  # 20-29 is young adult
        else:
            return self.age_groups[4]  # 30+ is adult
