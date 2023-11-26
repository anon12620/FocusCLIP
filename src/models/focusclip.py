from typing import List, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig, AutoModel, AutoTokenizer


class FocusCLIP(nn.Module):
    """
    Contrastive Language-Image Pre-training with Subject-Level Focus (FocusCLIP) extends CLIP by introducing an
    additional input modality, a region of interest (ROI) mask, to the model.

    The ROI mask is a heatmap of the image that highlights the region of interest, e.g. a human, in the image.
    FocusCLIP uses the ROI mask to learn a joint representation of the image and the ROI mask by aligning both
    the image and text features with the ROI mask features during training. This allows the model to learn a
    joint representation of the image, the ROI mask, and the text, which leads to better feature representations
    for the intended downstream tasks.

    FocusCLIP is a generalization of CLIP, which is a zero-shot learning model that learns a joint embedding space
    of images and text. The ROI input and the corresponding visual encoder are only used during training, making
    the inference of FocusCLIP identical to CLIP.

    This implementation uses visual models from the PyTorch Image Models (timm) library and text models from the
    HuggingFace Transformers library. These encoders are loaded without pretrained weights by default, following
    the original CLIP implementation. However, pretrained weights can be loaded by setting the `pretrained` flag
    to `True`. If the `triple_components` flag is set to `False`, the model is identical to CLIP.

    Args:
        visual_encoder_name (str): Name of the pretrained visual model.
        text_encoder_name (str): Name of the pretrained text model.
        tokenizer_name (str): Name of the pretrained tokenizer.
        context_length (int): Length of the text context.
        triple_components (bool): If True, the model uses the ROI mask and the masked image encoder.
        shared_encoder (bool): If True, the image and masked image encoders are shared. If triple_components is
                               False, this flag is ignored.
        pretrained (bool): If True, the pretrained weights are loaded.
    """

    def __init__(self,
                 visual_encoder_name: str = 'vit_base_patch16_224',
                 text_encoder_name: str = 'bert-base-uncased',
                 tokenizer_name: str = 'bert-base-uncased',
                 embed_dim: int = 512,
                 context_length: int = 128,
                 triple_components: bool = True,
                 shared_encoder: bool = False,
                 pretrained=True):
        super().__init__()
        self.context_length = context_length
        self.embed_dim = embed_dim

        # Image Encoder
        self.visual = timm.create_model(visual_encoder_name,
                                        pretrained=pretrained, num_classes=0)
        self.visual_embed = nn.Linear(self.visual.num_features, self.embed_dim)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Text Encoder
        if pretrained:
            self.text = AutoModel.from_pretrained(text_encoder_name)
        else:
            cfg = AutoConfig.from_pretrained(text_encoder_name)
            self.text = AutoModel.from_config(cfg)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.text_embed = nn.Linear(self.text.config.hidden_size, self.embed_dim)

        # Masked Image Encoder (for triplets)
        self.alpha = None
        self.visual_masked = None
        if triple_components:
            self.alpha = nn.Parameter(torch.tensor(0.5))
            if shared_encoder:
                self.visual_masked = self.visual
                self.visual_masked_embed = self.visual_embed
            else:
                self.visual_masked = timm.create_model(visual_encoder_name,
                                                       pretrained=pretrained,
                                                       num_classes=0)
                self.visual_masked_embed = nn.Linear(self.visual_masked.num_features, self.embed_dim)

    def tokenize(self, text: Union[str, List[str]]):
        return self.tokenizer(text, return_tensors='pt', padding='max_length',
                              truncation=True, max_length=self.context_length)

    def encode_image(self, image: Tensor):
        x = self.visual(image)
        x = self.visual_embed(x)
        return x

    def encode_text(self, text: Union[str, List]):
        tokens = self.tokenize(text)
        tokens = tokens.to(self.text.device)
        x = self.text(**tokens).last_hidden_state
        x = self.avgpool(x.transpose(1, 2)).squeeze(2)  # NLD -> NDL -> ND
        x = self.text_embed(x)
        return x

    def forward_features(self, image: Tensor, text: Union[str, List]):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        return image_features, text_features

    def forward_image_mask(self, image: Tensor, mask: Tensor = None):
        """ Forward pass for the masked image encoder.

        The mask is scaled by a learnable parameter alpha which decides the
        amount of background to be retained in the masked image. The masked
        image is then encoded by the visual encoder.

        Args:
            image (Tensor): Image tensor of shape (N, C, H, W).
            mask (Tensor): Mask tensor of shape (N, 1, H, W) or None.

        Returns:
            Tensor: Masked image features of shape (N, D) or None.
        """
        if mask is None:
            return None

        if self.alpha is None or self.visual_masked is None:
            return None

        # Expand mask channels to match image channels
        mask = mask.expand_as(image)

        # Scale mask by alpha and apply to image
        scaled_mask = self.alpha + mask * (1 - self.alpha)
        masked_image = image * scaled_mask

        # Encode masked image
        masked_features = self.visual_masked(masked_image)
        masked_features = self.visual_masked_embed(masked_features)
        return masked_features

    def forward(self, image: Tensor, text: Union[str, List], mask: Tensor = None):
        image_features, text_features = self.forward_features(image, text)
        masked_features = self.forward_image_mask(image, mask)
        return image_features, text_features, masked_features

    def inference(self, image: Tensor, text: Union[str, List]):
        image_features, text_features = self.forward_features(image, text)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        return (image_features @ text_features.T).softmax(dim=-1)[0]
