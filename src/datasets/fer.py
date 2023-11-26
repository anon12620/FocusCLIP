import os

from torchvision.datasets import FER2013 as TorchFER2013


class FER(TorchFER2013):
    """ The Facial Emotion Recognition (FER) 2013 dataset. """

    def __init__(self, root='data/fer2013', split='train', transform=None):
        super().__init__(root, split, transform)
        self.root = os.path.join(root, split)
        self.transform = transform

        self.emotion_names = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
        self.emotion_adjectives = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

    def __getitem__(self, idx):
        image, emotion_id = super().__getitem__(idx)
        image = image.convert('RGB')

        return image, {
            'emotion_id': emotion_id,
            'emotion_name': self.emotion_names[emotion_id],
            'emotion_adjective': self.emotion_adjectives[emotion_id],
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