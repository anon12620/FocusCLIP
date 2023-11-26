import json
import os


from .face import AgeGenderRaceDataset


class EmoticDataset(AgeGenderRaceDataset):
    def __init__(self, root, split='train', **kwargs):
        assert split in ['train']
        super().__init__(root, split, **kwargs)

    @property
    def num_emotions(self):
        return len(self.emotions)

    def caption_emotion(self, emotion):
        assert emotion in self.emotions, f'emotion {emotion} not in {self.emotions}'
        return f'a photo of a person who is feeling {emotion.lower()}'

    @property
    def captions_emotion(self):
        return [self.caption_emotion(e) for e in self.emotions]

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
        self.age_groups = sorted(list(set([s[1]['age_group'] for s in samples])))
        self.genders = sorted(list(set([s[1]['gender'] for s in samples])))

        # Add class IDs to the samples
        for sample in samples:
            sample[1]['emotion_id'] = self.emotions.index(sample[1]['emotion'])
            sample[1]['age_group_id'] = self.age_groups.index(sample[1]['age_group'])
            sample[1]['gender_id'] = self.genders.index(sample[1]['gender'])

        return samples

    def _transform_image(self, image, target):
        """ Apply the image transform, if specified. """
        # Crop the person from the image
        x1, y1, x2, y2 = tuple(target['bbox'])
        image = image.crop((x1, y1, x2, y2))

        # Apply the transform if specified
        return self._transform(image) if self._transform else image


if __name__ == '__main__':
    data = EmoticDataset()
    print('Number of samples:', len(data))
    print('Number of emotions:', len(data.emotions))
    print('Emotions:', data.emotions)

    # Iterate over the whole dataset, and calculate emotion, age and gender distributions
    emotion_dist = {k: {
        'Male': {
            'Adult': 0,
            'Kid': 0,
            'Teenager': 0
        },
        'Female': {
            'Adult': 0,
            'Kid': 0,
            'Teenager': 0
        }
    } for k in data.emotions}

    for image, annotation in data:
        emotion = annotation['emotion']
        gender = annotation['gender']
        age = annotation['age']

        emotion_dist[emotion][gender][age] += 1

    with open('emotion_dist.json', 'w') as f:
        json.dump(emotion_dist, f, indent=4)
    print('Emotion distribution:', emotion_dist)
