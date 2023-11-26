from typing import List, Union

import torch.nn as nn
from torch import Tensor
from transformers import CLIPModel, CLIPProcessor


class CLIP(nn.Module):
    """
    The original CLIP model.

    CLIP is a zero-shot learning model that learns a joint embedding space of images and text.
    This implementation is a wrapper around the CLIP models from HuggingFace Transformers, and
    is used for finetuning the pretrained CLIP models on our dataset.

    The following models are supported:
    - clip-vit-base-patch16
    - clip-vit-large-patch16
    - clip-vit-large-patch14
    - clip-vit-large-patch14-336

    Args:
        model_name (str): Name of the pretrained CLIP model.
    """

    def __init__(self, variant: str = 'clip-vit-base-patch16'):
        super().__init__()
        supported_models = [
            'clip-vit-base-patch16',
            'clip-vit-large-patch16',
            'clip-vit-large-patch14',
            'clip-vit-large-patch14-336'
        ]
        assert variant in supported_models, \
            f'Model name must be one of {supported_models} but got {variant}'

        self.variant = f'openai/{variant}'

        self.model = CLIPModel.from_pretrained(self.variant)
        self.processor = CLIPProcessor.from_pretrained(self.variant)

    def forward_features(self, image: Tensor, text: Union[str, List]):
        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(image.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.image_embeds, outputs.text_embeds

    def forward(self, image: Tensor, text: Union[str, List], ignored: Tensor = None):
        image_features, text_features = self.forward_features(image, text)
        masked_features = None
        return image_features, text_features, masked_features

    def inference(self, image: Tensor, text: Union[str, List]):
        return self.forward_features(image, text)
