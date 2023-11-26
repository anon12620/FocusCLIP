import os
from typing import Callable, Optional

import pandas as pd

from PIL import Image
from torch.utils.data import Dataset


class FERPlus(Dataset):
    """ The FER+ dataset [1] is an extension of the FER2013 dataset [2] with improved annotations.

    References:
        [1] https://github.com/Microsoft/FERPlus
        [2] https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
    """

    def __init__(self, root='data/emotion-recognition/ferplus', split='train',
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        assert split in ['train', 'val', 'test']
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Define the emotions
        self.emotion_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
        self.emotion_adjectives = ['neutral', 'happy', 'surprised', 'sad', 'angry', 'disgusted', 'afraid', 'contemptuous']

        # Load the annotations
        self.annotations = pd.read_csv(os.path.join(self.root, 'annotations', f'{split}.csv'), header=None)
        self.annotations.columns = ['image_id', 'face_bbox'] + self.emotion_names + ['unknown', 'NF']

        # Remove images with no face (NF>0) and the NF column
        self.annotations = self.annotations[self.annotations['NF'] == 0]
        self.annotations = self.annotations.drop(columns=['NF'])

        # Convert dataframe to dictionary
        self.annotations = self.annotations.to_dict(orient='records')

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        # Load the image
        image_path = os.path.join(self.root, 'images', self.split, annotation['image_id'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image) if self.transform else image

        # Read the emotion scores and convert them to probabilities
        emotion_scores = [float(annotation[emotion]) for emotion in self.emotion_names]
        emotion_probs = [score / sum(emotion_scores) for score in emotion_scores]

        # Apply the target transform if specified
        if self.target_transform:
            emotion_probs = self.target_transform(emotion_probs)

        # Get id of the emotion with the highest probability
        emotion_id = emotion_probs.index(max(emotion_probs))

        return image, {
            'emotion_id': emotion_id,
            'emotion_name': self.emotion_names[emotion_id],
            'emotion_adjective': self.emotion_adjectives[emotion_id],
            # 'emotion_probs': emotion_probs,
        }

    def make_prompt(self, emotion):
        assert emotion in self.emotion_adjectives
        article = 'a'
        if emotion[0].lower() in ['a', 'e', 'i', 'o', 'u']:
            article = 'an'
        return f'a photo of {article} {emotion} looking face'
    
    @property
    def prompts(self):
        return [self.make_prompt(emotion) for emotion in self.emotion_adjectives]

    @property
    def num_emotions(self):
        return len(self.emotion_names)
    
    @property
    def captions_emotion(self):
        return self.prompts