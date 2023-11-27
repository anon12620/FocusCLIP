import json
import os

from ._base import AgeGenderDataset


class EmoticDataset(AgeGenderDataset):
    """ The EMOTIC dataset contains person images with age, gender and emotion annotations. """

    def __init__(self, root='data/eval/emotic', split='train', **kwargs):
        assert split in ['train']
        super().__init__(root, split, **kwargs)
        assert hasattr(self, 'emotions'), 'self.emotions must be defined in _load_split_data'
        self.num_emotions = len(self.emotions)  # type: ignore
        self.captions_emotion = [self._map_emotion(e) for e in self.emotions]  # type: ignore

    def _map_emotion(self, emotion):
        assert emotion in self.emotions, f'emotion {emotion} not in {self.emotions}'
        return f'a photo of a person who is feeling {emotion.lower()}'

    def _load_split_data(self, split):
        # Read the annotations
        with open(os.path.join(self._root, f'annotations_{split}.json'), 'r') as f:
            annotations = json.load(f)

        # Create the samples
        samples = []
        for annotation in annotations:
            image_name = annotation['image_name']
            image_folder = annotation['image_folder']
            emotion_data = annotation['emotion_data']

            image_path = os.path.join(self._root, image_folder, image_name)
            for entry in emotion_data:
                samples.append((image_path, {
                    'bbox': entry['bbox'],
                    'emotion': entry['categories'],
                    'emotion_valence': entry['valence'],
                    'emotion_arousal': entry['arousal'],
                    'emotion_dominance': entry['dominance'],
                    'gender': entry['gender'],
                    'age_group': entry['age'],
                }))

        # Get class names
        self.emotions = sorted(list(set([s[1]['emotion'] for s in samples])))
        self.age_groups = sorted(
            list(set([s[1]['age_group'] for s in samples])))
        self.genders = sorted(list(set([s[1]['gender'] for s in samples])))

        # Add class IDs to the samples
        for sample in samples:
            sample[1]['emotion_id'] = self.emotions.index(sample[1]['emotion'])
            sample[1]['age_group_id'] = self.age_groups.index(
                sample[1]['age_group'])
            sample[1]['gender_id'] = self.genders.index(sample[1]['gender'])

        return samples

    def _transform_image(self, image, target):
        """ Apply the image transform, if specified. """
        # Crop the person from the image
        x1, y1, x2, y2 = tuple(target['bbox'])
        image = image.crop((x1, y1, x2, y2))

        # Apply the transform if specified
        return self._transform(image) if self._transform else image
