import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses


class NTXentLoss(nn.Module):
    """ Self-Supervisied Normalized Temperature-Scaled Cross-Entropy (NT-Xent) Loss.

    This is a self-supervised loss function that is used to train a model with image-text
    pairs without any labels. It is based on the cross-entropy loss function, and is
    computed by comparing the similarity between all pairs of image and text embeddings,
    and maximizing the similarity between positive pairs and minimizing the similarity
    between negative pairs.

    The similarity between two embeddings is computed by taking the cosine similarity
    between them. The similarity matrix is then normalized by the temperature parameter
    and fed to the softmax function to obtain a probability distribution. The loss is
    then computed by taking the negative log-likelihood of the positive pairs.

    The temperature parameter is a hyperparameter which scales the logits before they
    are fed to the softmax function, and controls the sharpness of the probability
    distribution. A higher temperature results in a softer probability distribution, and
    a lower temperature results in a sharper probability distribution. This parameter
    can be learned by setting the `learn_temperature` parameter to `True`. By default,
    the temperature parameter is set to 0.5 and is not learned.

    Args:
        temperature (float): The temperature parameter that scales the logits.
        learn_temperature (bool): Whether to learn the temperature parameter.
    """

    def __init__(self, temperature=0.5, learn_temperature=False):
        super().__init__()
        if learn_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.temperature = temperature

        criteria = losses.NTXentLoss(temperature=self.temperature)
        self.criteria = losses.SelfSupervisedLoss(criteria)

    def compute_similarity(self, image_embeddings, text_embeddings):
        """ Computes pairwise cosine similarity between the image and text embeddings.

        Args:
            image_embeddings (torch.Tensor): The image embeddings of shape (B, embed_dim).
            text_embeddings (torch.Tensor): The text embeddings of shape (B, embed_dim).

        Returns:
            torch.Tensor: The similarity matrix of shape (B, B).
        """
        return F.cosine_similarity(image_embeddings.unsqueeze(1),
                                   text_embeddings.unsqueeze(0),
                                   dim=-1)

    def forward(self, image_embeddings, text_embeddings):
        """ Computes the NT-Xent loss between the image and text embeddings.

        Args:
            image_embeddings (torch.Tensor): The image embeddings of shape (B, embed_dim).
            text_embeddings (torch.Tensor): The text embeddings of shape (B, embed_dim).

        Returns:
            torch.Tensor: The NT-Xent loss of shape (1).
        """
        return self.criteria(image_embeddings, text_embeddings)
