import json
import os

from PIL import Image
from torch.utils.data import Dataset


class MPIIDataset(Dataset):
    def __init__(self, root, split='train', transforms=None):
        assert split in ['train', 'val'], \
            "Split must be either 'train' or 'val'"

        self.root = root
        self.split = split
        self.transforms = transforms if transforms is not None else lambda x: x

        self.annos = self._load_annotations()
        self.activities = self._get_activities()
        self.num_activities = len(self.activities)

    def _load_annotations(self):
        annoFile = os.path.join(self.root, 'annotations', f'{self.split}.json')
        with open(annoFile, "r") as f:
            annos = json.load(f)

        if self.split == 'val':
            annos = [anno for anno in annos if anno['istrain'] == False]

        return annos

    def _get_activities(self):
        activities = {}
        for anno in self.annos:
            activity_id = anno['activity_id']
            activity_name = anno['activity'].lower().strip()
            if activity_id not in activities:
                activities[activity_id] = activity_name
        return activities

    def get_activity_name(self, activity_id):
        return self.activities[activity_id]

    def get_activity_names(self):
        return list(set(self.activities.values()))

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        item = self.annos[idx]
        # 'image', 'video_id', 'video_frame', 'activity_id',
        # 'activity', 'count', 'people', 'istrain', 'description'

        # Load image
        image_path = os.path.join(self.root, 'images', item['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)

        # Load mask
        mask_path = os.path.join(
            self.root, 'body_part_masks', item['image'].replace('.jpg', '.png'))
        mask = Image.open(mask_path).convert('L')
        mask = self.transforms(mask)

        # Parse people annotations
        # 'id', 'center', 'scale', 'kpts', 'kpts_vis'
        people = item['people']
        detections = []
        for person in people:
            kpts = person['kpts']  # list of [x, y]
            kpts_vis = person['kpts_vis']  # list of 0 or 1
            kpts = [kpt + [v] for kpt, v in zip(kpts, kpts_vis)]

            center = person['center']  # [x, y]
            scale = person['scale']  # float
            size = scale * 200  # 200 is a constant for MPII
            x_min = center[0] - size / 2
            y_min = center[1] - size / 2
            width = size
            height = size  # assuming a square bounding box
            bbox = [x_min, y_min, width, height]

            detection = {
                'id': person['id'],
                'keypoints': kpts,
                'bbox': bbox,
            }
            detections.append(detection)

        # Create label
        labels = {
            'video_id': item['video_id'],
            'video_frame': item['video_frame'],
            'activity_id': item['activity_id'],
            'activity': item['activity'],
            'num_people': item['count'],
            'detected_people': detections,
            'caption': item['description'],
            'mask': mask,
        }

        return image, labels
