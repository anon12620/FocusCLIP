import os

import pandas as pd

from ._base import AgeGenderRaceDataset


class LAGENDADataset(AgeGenderRaceDataset):
    """ The LAGENDA dataset contains person images with age, gender and race annotations."""

    def __init__(self, root='data/eval/lagenda', split='face', **kwargs):
        assert split in ['face', 'person', 'none']  # none means images will not be cropped
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
