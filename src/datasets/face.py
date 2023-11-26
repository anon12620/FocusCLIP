import os

import pandas as pd

from .base import BaseDataset


class AgeGenderRaceDataset(BaseDataset):
    def __init__(self, root, split='train', **kwargs):
        super().__init__(root, split, **kwargs)

    @property
    def num_age_groups(self):
        return len(self.age_groups)

    @property
    def num_races(self):
        return len(self.races)

    @property
    def num_genders(self):
        return len(self.genders)

    def caption_age(self, age_group):
        assert age_group in self.age_groups, f'age {age_group} not in {self.age_groups}'
        return f'a photo of a {age_group.lower()} person'

    def caption_race(self, race):
        assert race in self.races, f'race {race} not in {self.races}'
        if race == 'others':
            race = 'other race'
        if race[0] in ['a', 'e', 'i', 'o', 'u']:
            article = 'an'
        else:
            article = 'a'
        return f'a photo of {article} {race.lower()} person'

    def caption_gender(self, gender):
        assert gender in self.genders, f'gender {gender} not in {self.genders}'
        return f'a photo of a {gender.lower()} person'

    def caption_age_and_gender(self, age_group, gender):
        assert age_group in self.age_groups, f'age {age_group} not in {self.age_groups}'
        assert gender in self.genders, f'gender {gender} not in {self.genders}'
        return f'a photo of a {gender.lower()} {age_group.lower()} person'

    @property
    def captions(self):
        _captions = []
        for a in self.age_groups:
            for g in self.genders:
                _captions.append(self.caption_age_and_gender(a, g))
        return _captions

    @property
    def captions_age(self):
        return [self.caption_age(a) for a in self.age_groups]

    @property
    def captions_race(self):
        return [self.caption_race(r) for r in self.races]

    @property
    def captions_gender(self):
        return [self.caption_gender(g) for g in self.genders]


class FairFace(AgeGenderRaceDataset):
    """ The FairFace dataset. """

    def __init__(self, root, split='train', **kwargs):
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


class UTKFace(AgeGenderRaceDataset):
    """ The UTKFace dataset. """

    def __init__(self, root, split='train', **kwargs):
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


class LAGENDA(AgeGenderRaceDataset):
    def __init__(self, root, split='face', **kwargs):
        assert split in ['face', 'person', 'none']
        super().__init__(root, split, **kwargs)

    def _load_split_data(self, split):
        # Read the annotations
        anno_file = os.path.join(self._root, 'annotations.csv')
        df = pd.read_csv(anno_file)  # file,age,gender,race,service_test

        # Remove all rows where any column is -1
        df = df[(df != -1).all(1)]

        # Update img_name by splitting / and keeping last part only
        df = df.assign(img_name=df['img_name'].str.split('/').str[-1])

        self.age_groups = ['kid', 'teenager', 'adult']
        self.genders = ['male', 'female']

        # Filter out invalid images
        images_dir = os.path.join(self._root, 'images')
        images = [n.split('/')[-1] for n in df['img_name']]
        images = [i for i in images if os.path.exists(
            os.path.join(images_dir, i))]

        # Create the samples
        samples = []
        for item in df.to_dict('records'):
            image_name = item['img_name'].split('/')[-1]
            if image_name not in images:
                continue

            age = item["age"]
            if age <= 12:
                age_group = 'kid'
            elif age <= 19:
                age_group = 'teenager'
            else:
                age_group = 'adult'

            gender = item['gender']
            if gender == 'M':
                gender = 'male'
            else:
                gender = 'female'

            # Get face bounding box
            face_x0 = item['face_x0']
            face_y0 = item['face_y0']
            face_x1 = item['face_x1']
            face_y1 = item['face_y1']
            face_bbox = [face_x0, face_y0, face_x1, face_y1]

            # Get person bounding box
            person_x0 = item['person_x0']
            person_y0 = item['person_y0']
            person_x1 = item['person_x1']
            person_y1 = item['person_y1']
            person_bbox = [person_x0, person_y0, person_x1, person_y1]

            bbox = person_bbox if split == 'person' \
                else face_bbox if split == 'face' \
                else None

            image_path = os.path.join(images_dir, image_name)
            samples.append((image_path, {
                'age': -1,
                'age_group': age_group,
                'age_group_id': self.age_groups.index(age_group),
                'gender': gender,
                'gender_id': self.genders.index(gender),
                'bbox': bbox,
            }))
        return samples

    def _transform_image(self, image, target):
        """ Apply the image transform, if specified. """
        # Crop the image to required bounding box, if specified
        bbox = target['bbox']
        if bbox is not None:
            x0, y0, x1, y1 = tuple(bbox)
            image = image.crop((x0, y0, x1, y1))

        # Apply the transform if specified
        return self._transform(image) if self._transform else image