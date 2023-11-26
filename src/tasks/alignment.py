import io

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image

from losses import get_loss
from models import get_model
from utils.viz import visualize_with_pca, visualize_with_tsne


class AlignmentTask(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_model(cfg.MODEL.NAME, **cfg.MODEL.KWARGS)
        self.loss_fn = get_loss(cfg.LOSS.NAME, **cfg.LOSS.KWARGS)

        self.confusion_matrix_list = []

    def forward(self, batch):
        image, mask, text = tuple(batch)
        return self.model(image, text, mask)

    def _step(self, batch, batch_idx, mode='train'):
        # Forward pass
        outputs = self.forward(batch)
        image_features, text_features, masked_features = outputs
        if masked_features is None:
            loss = self.loss_fn(image_features, text_features)
        else:
            loss = self.loss_fn(masked_features, image_features) + \
                self.loss_fn(masked_features, text_features) + \
                self.loss_fn(image_features, text_features)

        if mode == 'val':
            self.accumulate_confusion(image_features, text_features)

        if batch_idx == 0:
            labels = ['Images', 'Texts']
            embeddings = [image_features, text_features]
            if masked_features is not None:
                labels.append('ROI Masks')
                embeddings.append(masked_features)

            pca = visualize_with_pca(embeddings, labels)
            self.logger.experiment.add_image(
                f'plot/pca_{mode}', pca, self.global_step)

            tsne = visualize_with_tsne(embeddings, labels)
            self.logger.experiment.add_image(
                f'plot/tsne_{mode}', tsne, self.global_step)

        return loss

    def forward_features(self, batch):
        image, meta = tuple(batch)
        text = meta['caption']
        return self.model.forward_features(image, text)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, mode='train')
        self.log('loss/train', loss, prog_bar=True,
                 batch_size=batch[0].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, mode='val')
        self.log('loss/val', loss, prog_bar=True, batch_size=batch[0].shape[0])
        return loss

    def on_validation_epoch_end(self):
        # ignore last batch (because it might not be full)
        self.confusion_matrix_list = self.confusion_matrix_list[:-1]

        # Compute the average confusion matrix
        avg_confusion_matrix = torch.stack(
            self.confusion_matrix_list).mean(dim=0)
        self.show_confusion_matrix(avg_confusion_matrix)
        self.confusion_matrix_list = []

    def accumulate_confusion(self, image_features, text_features):
        with torch.no_grad():
            # Move the features to CPU
            image_features = image_features.detach().cpu()
            text_features = text_features.detach().cpu()

            # Compute the similarity matrix
            similarity_matrix = self.loss_fn.compute_similarity(
                image_features, text_features)

            # Accumulate the confusion matrix (take running average)
            self.confusion_matrix_list.append(similarity_matrix)

    def show_confusion_matrix(self, similarity_matrix):
        batch_size = similarity_matrix.shape[0]

        # Create the figure and plot the confusion matrix
        plt.figure(figsize=(20, 14))
        plt.imshow(similarity_matrix)

        # Plot the similarity scores in the matrix cells
        for x in range(similarity_matrix.shape[1]):
            for y in range(similarity_matrix.shape[0]):
                plt.text(
                    x, y, f"{similarity_matrix[y, x]:.2f}", ha="center", va="center", size=12)

        for side in ["left", "top", "right", "bottom"]:
            plt.gca().spines[side].set_visible(False)

        # Calculate the bottom y-axis limit dynamically
        bottom_limit = -1.5  # Adjust this value as needed
        plt.xlim([-0.5, batch_size - 0.5])
        plt.ylim([batch_size + 0.5, bottom_limit])

        # Computer average similarity on the diagonal
        avg_similarity = similarity_matrix.diag().mean().item()

        # Computer average similarity on the off-diagonal
        off_diagonal = similarity_matrix - torch.diag(similarity_matrix.diag())
        avg_off_diagonal_similarity = off_diagonal.sum() / (batch_size * (batch_size - 1))

        plt.title("Cosine similarity between image and text embeddings\n"
                  f"Average similarity between positive pairs: {avg_similarity:.2f}\n"
                  f"Average similarity between negative pairs: {avg_off_diagonal_similarity:.2f}",
                  size=16)
        plt.tight_layout()

        # Read confusion image (as numpy array) and log to tensorboard
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        confusion = np.asarray(Image.open(buf))
        confusion = torch.tensor(confusion).permute(2, 0, 1)
        self.logger.experiment.add_image('confusion/val', confusion,
                                         self.global_step)
        buf.close()
        plt.close()

        # Delete  local variables to free up memory
        del similarity_matrix, confusion

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(),
                               lr=self.hparams.cfg.TRAIN.LR,
                               momentum=self.hparams.cfg.TRAIN.MOMENTUM)
