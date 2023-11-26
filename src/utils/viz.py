import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def draw_keypoints(im, keypoints, color=(0, 255, 0)):
    """ Draw keypoints on image and return image.

    Args:
        im (np.ndarray): Image to draw keypoints on (H, W, C).
        keypoints (np.ndarray): Keypoints to draw (N, 3) where N is the number of keypoints and 3 is (x, y, c) where c is the confidence score.
    """

    for i, (x, y, score) in enumerate(keypoints):
        # skip if (0, 0)
        if (x == 0 and y == 0) or score < 0.5:
            continue

        cv2.circle(im, (int(x), int(y)), 2, color, -1)
        # cv2.putText(im, str(i), (int(x), int(y)),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return im


def draw_line(im, keypointA, keypointB, color=(0, 255, 0)):
    """ Draw line between two keypoints on image and return image. """
    # skip if either is invalid
    scoreA = keypointA[2]
    scoreB = keypointB[2]
    if scoreA <= 0.5 or scoreB <= 0.5:
        return im

    cv2.line(im, (int(keypointA[0]), int(keypointA[1])),
             (int(keypointB[0]), int(keypointB[1])), color, 2)

    return im


def draw_skeleton(im, keypoints, color=(0, 255, 0)):
    """ Draw skeleton on image and return image. """
    im = draw_line(im, keypoints[0], keypoints[1], color)
    im = draw_line(im, keypoints[1], keypoints[2], color)
    im = draw_line(im, keypoints[2], keypoints[6], color)
    im = draw_line(im, keypoints[6], keypoints[3], color)
    im = draw_line(im, keypoints[3], keypoints[4], color)
    im = draw_line(im, keypoints[4], keypoints[5], color)
    im = draw_line(im, keypoints[6], keypoints[7], color)
    im = draw_line(im, keypoints[7], keypoints[8], color)
    im = draw_line(im, keypoints[8], keypoints[9], color)
    im = draw_line(im, keypoints[7], keypoints[12], color)
    im = draw_line(im, keypoints[7], keypoints[13], color)
    im = draw_line(im, keypoints[12], keypoints[11], color)
    im = draw_line(im, keypoints[11], keypoints[10], color)
    im = draw_line(im, keypoints[13], keypoints[14], color)
    im = draw_line(im, keypoints[14], keypoints[15], color)

    return im


def draw(im, keypoints, color):
    """ Draw keypoints and skeleton on image and return image. """

    # copy image to draw on
    im = im.copy()
    im = draw_skeleton(im, keypoints, (1, color[1], color[2]))
    im = draw_keypoints(im, keypoints, (0, color[1], color[2]))

    return im


def show(im, gt_kpt, pd_kpt):
    """ Draw keypoints and skeleton on image and show image. """

    fig = []
    for i in range(min(im.shape[0], 4)):
        x = im[i].detach().cpu().numpy().transpose(1, 2, 0) / 255
        y = gt_kpt[i].detach().cpu().numpy()
        y_hat = pd_kpt[i].detach().cpu().numpy()

        row = np.hstack((draw(x.copy(), y, color=(0, 0, 1)),
                         draw(x.copy(), y_hat, color=(0, 1, 0))))
        fig.append(row)

    err = np.vstack(fig)
    return err


def visualize_with_pca(embeddings, labels):
    pca = PCA(n_components=2)
    embeddings = [emb.detach().cpu().numpy() for emb in embeddings]
    all_data = np.concatenate(embeddings, axis=0)
    pca_result = pca.fit_transform(all_data)

    # Plot the results
    for i, emb in enumerate(embeddings):
        plt.scatter(pca_result[:len(emb), 0], pca_result[:len(emb), 1], label=labels[i])
        pca_result = pca_result[len(emb):]

    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    fig = np.asarray(Image.open(buf))
    fig_as_tensor = torch.tensor(fig).permute(2, 0, 1)

    del pca_result, all_data, fig
    return fig_as_tensor


def visualize_with_tsne(embeddings, labels, perplexity=5, n_iter=300):
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
    embeddings = [emb.detach().cpu().numpy() for emb in embeddings]
    all_data = np.concatenate(embeddings, axis=0)
    tsne_results = tsne.fit_transform(all_data)

    # Plot the results
    for i, emb in enumerate(embeddings):
        plt.scatter(tsne_results[:len(emb), 0], tsne_results[:len(emb), 1], label=labels[i])
        tsne_results = tsne_results[len(emb):]

    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    fig = np.asarray(Image.open(buf))
    fig_as_tensor = torch.tensor(fig).permute(2, 0, 1)

    del tsne_results, all_data, fig
    return fig_as_tensor
