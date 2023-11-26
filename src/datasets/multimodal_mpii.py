import json
import os

from PIL import Image
from torch.utils.data import Dataset


def load_dataset(captions_dir, config_name, split='train'):
    filemap = {
        'train': {
            'gpt3.5-turbo-legacy': 'gpt-3.5-turbo-0301/train.json',
            'gpt3.5-turbo': 'gpt-3.5-turbo-0613/train.json',
            'llama-2': 'llama-2-70b-chat-hf/train.json',
            'gpt-4': 'gpt-4-0613/train.json',
            'dummy': 'dummy.json',
        },
        'validation': {
            'gpt3.5-turbo-legacy': 'gpt-3.5-turbo-0301/val.json',
            'gpt3.5-turbo': 'gpt-3.5-turbo-0613/val.json',
            'llama-2': 'llama-2-70b-chat-hf/val.json',
            'gpt-4': 'gpt-3.5-turbo-0613/val.json',
            'dummy': 'gpt-3.5-turbo-0301/val.json',
        },
    }

    split_file = os.path.join(captions_dir, filemap[split][config_name])
    with open(split_file, 'r') as f:
        print(f'Loading {split_file}...')
        data = json.load(f)
    return data


class MultiModalMPII(Dataset):
    def __init__(self, root='data/mpii', config_name='gpt3.5-turbo-legacy',
                 split='train', transforms=None):
        self.root = root
        self.img_dir = os.path.join(root, 'images')
        self.captions_dir = os.path.join(root, 'captions')
        self.masks_dir = os.path.join(root, 'heatmaps')
        self.split = split
        self.transforms = transforms

        if isinstance(config_name, str):
            config_name = [config_name]
        self.num_views = len(config_name)

        # Create an index
        index = {}
        for name in config_name:
            print(f'Loading {name} {split} dataset...')
            cfg_data = load_dataset(self.captions_dir, name, split=split)
            for item in cfg_data:
                image = item['image']
                caption = item['description']

                if image not in index:
                    index[image] = []

                index[image].append(caption)

        # Filter out invalid samples
        max_captions = max(len(captions) for captions in index.values())
        for image, captions in list(index.items()):
            # Remove images with less captions than the max (because we
            # want all images to have the same number of captions)
            if len(captions) < max_captions:
                del index[image]

            # Remove images that don't exist
            if not os.path.exists(os.path.join(self.img_dir, image)):
                del index[image]

            # Remove images that don't have a mask
            mask_name = image.replace('.jpg', '.png')
            if not os.path.exists(os.path.join(self.masks_dir, mask_name)):
                del index[image]

        self.images = [os.path.join(self.img_dir, image)
                       for image in index.keys()]
        self.masks = [os.path.join(self.masks_dir, image.replace('.jpg', '.png'))
                      for image in index.keys()]
        self.captions = list(index.values())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load the image
        image = self.images[idx]
        image = Image.open(image).convert('RGB')

        # Load the mask
        mask = self.masks[idx]
        mask = Image.open(mask).convert('L')

        # Load the captions
        captions = self.captions[idx]

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return image, mask, captions
