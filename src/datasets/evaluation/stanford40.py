import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Stanford40Dataset(Dataset):
    ALLOWED_SPLITS = ['train', 'test']

    def __init__(self, root='data/eval/stanford_actions', split='train', transform=None):
        super().__init__()
        if split not in self.ALLOWED_SPLITS:
            raise ValueError(f"Invalid split: {split}")

        split_files = self.load_split(root, split)
        activities = self.load_activities(root)
        self.activities = activities
        self.num_activities = len(activities)

        images_dir = os.path.join(root, 'JPEGImages')
        labels_dir = os.path.join(root, 'XMLAnnotations')

        self.data = []
        for file_name in split_files:
            image_path = os.path.join(images_dir, file_name)
            if not os.path.exists(image_path):
                continue

            label_file = os.path.join(labels_dir, file_name.replace('.jpg', '.xml'))
            if not os.path.exists(label_file):
                continue

            label = self.load_label(label_file)
            label['activity_id'] = activities.index(label['activity'])
            label['image_path'] = image_path
            self.data.append(label)

        if not self.data:
            raise ValueError("No matching images and labels found.")

        self.transform = transform if transform is not None else lambda x: x

    def get_activity_names(self):
        return list(set(self.activities))

    @staticmethod
    def load_activities(root):
        activities_path = os.path.join(root, 'ImageSplits', 'actions.txt')
        activities = np.loadtxt(activities_path, dtype=str).tolist()
        activities = [activity[0] for activity in activities[1:]]
        return activities

    @staticmethod
    def load_split(root, split):
        splits_path = os.path.join(root, 'ImageSplits', f'{split}.txt')
        split_files = np.loadtxt(splits_path, dtype=str).tolist()
        return split_files

    @staticmethod
    def load_label(path):
        tree = ET.parse(path)
        root = tree.getroot()
        
        filename = root.find('filename').text
        activity = root.find('object/action').text
        xmin = int(root.find('object/bndbox/xmin').text)
        ymin = int(root.find('object/bndbox/ymin').text)
        xmax = int(root.find('object/bndbox/xmax').text)
        ymax = int(root.find('object/bndbox/ymax').text)
        
        # Converting to COCO format [x, y, width, height]
        bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
        
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        
        return {
            'image_id': filename,
            'activity': activity,
            'bbox': bbox,
            'width': width,
            'height': height
        }

    def __len__(self):
        return len(self.data)

    def get_activity(self, idx):
        return self.activities[idx]
    
    def __getitem__(self, idx):
        label = self.data[idx]
        image_path = label['image_path']
        image = Image.open(image_path)
        image = self.transform(image)
        return image, label

    @property
    def activity_captions(self):
        return [
            'a photo of a person {}'.format(name.replace('_', ' ').lower())
            for name in self.get_activity_names()
        ]