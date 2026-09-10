import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Optional


def tsne_plot(
    embeddings: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    anchors_mask: Optional[torch.Tensor] = None,
    alpha: float = 0.4,
    anchor_alpha: float = 0.9,
    figsize=(6, 6),
):
    """
    Visualize embeddings with t-SNE.

    Args:
        embeddings: (N, D) tensor
        labels: optional (N,) tensor for coloring
        anchors_mask: optional boolean mask for anchor points
        perplexity: t-SNE perplexity
        n_iter: t-SNE iterations
        alpha: transparency for non-anchor points
        anchor_alpha: transparency for anchor points
    """

    X = embeddings.detach().cpu().numpy()

    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        random_state=0,
    )
    Z = tsne.fit_transform(X)

    plt.figure(figsize=figsize)

    if labels is None:
        plt.scatter(Z[:, 0], Z[:, 1], s=10, alpha=alpha)
    else:
        labels_np = labels.detach().cpu().numpy()
        plt.scatter(
            Z[:, 0],
            Z[:, 1],
            c=labels_np,
            s=10,
            alpha=alpha,
            cmap="tab10",
        )

    if anchors_mask is not None:
        mask = anchors_mask.detach().cpu().numpy()
        plt.scatter(
            Z[mask, 0],
            Z[mask, 1],
            s=40,
            edgecolors="black",
            linewidths=0.8,
            alpha=anchor_alpha,
        )

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()
