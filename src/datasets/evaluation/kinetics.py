import os
import json


from ._base import BaseDataset


class KineticsDataset(BaseDataset):

    def __init__(self, root='data/eval/kinetcis400', split='train', **kwargs):
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}"
        super().__init__(root, split, **kwargs)

    def _load_split_data(self, split) -> list:
        # Load the annotations
        anno_file = os.path.join(self._root, 'annotations', f'{split}.json')
        with open(anno_file, 'r') as f:
            annotations = json.load(f)

        # Filter out annotations with missing images
        self.activities = []
        images_dir = os.path.join(self._root, 'images', self._split)
        samples = []
        for image_id, annotation in annotations.items():
            activity = annotation['annotations']['label']
            start = int(annotation['annotations']['segment'][0])
            end = int(annotation['annotations']['segment'][1])

            image_path = os.path.join(images_dir, activity, f'{image_id}_{start:06d}_{end:06d}.jpg')
            if os.path.exists(image_path):
                self.activities.append(activity)

                samples.append((image_path, {
                    'image_id': image_id,
                    'activity': activity,
                }))

        # Get unique activities
        self.activities = list(set(self.activities))

        # Add activity id to the samples
        for sample in samples:
            sample[1]['activity_id'] = self.activities.index(sample[1]['activity'])

        return samples

    @property
    def num_activities(self):
        return len(self.activities)

    def caption_activity(self, activity):
        assert activity in self.activities, f"Invalid activity: {activity}"
        return f'a photo of a person {activity}'
    
    @property
    def activity_captions(self):
        return [self.caption_activity(a) for a in self.activities]
