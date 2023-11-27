import os

import pandas as pd

from ._base import AgeGenderRaceDataset


class FairFaceDataset(AgeGenderRaceDataset):
    """ The FairFace dataset contains facial images with age, gender, and race annotations. """

    def __init__(self, root='data/eval/fairface', split='train', **kwargs):
        assert split in ['train', 'val']
        super().__init__(root, split, **kwargs)

    def _load_split_data(self, split):
        # Read the annotations
        anno_file = os.path.join(self._root, 'annotations', f'{split}.csv')
        df = pd.read_csv(anno_file)  # file,age,gender,race,service_test
        df = df.drop(columns=['service_test'])

        self.age_groups = sorted(df['age'].unique().tolist())
        self.age_groups = [f'{a} years old' for a in self.age_groups]
        self.genders = sorted(df['gender'].unique().tolist())
        self.races = sorted(df['race'].unique().tolist())

        # Filter out invalid images
        images_dir = os.path.join(self._root, 'images', split)
        images = [n.split('/')[-1] for n in df['file']]
        images = [i for i in images if os.path.exists(
            os.path.join(images_dir, i))]

        # Create the samples
        samples = []
        for item in df.to_dict('records'):
            image_name = item['file'].split('/')[-1]
            if image_name not in images:
                continue

            age_group = f'{item["age"]} years old'
            gender = item['gender']
            race = item['race']

            image_path = os.path.join(images_dir, image_name)
            samples.append((image_path, {
                'age': -1,
                'age_group': age_group,
                'age_group_id': self.age_groups.index(age_group),
                'gender': gender,
                'gender_id': self.genders.index(gender),
                'race': race,
                'race_id': self.races.index(race),
            }))
        return samples
