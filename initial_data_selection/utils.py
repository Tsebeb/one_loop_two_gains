import copy

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Tuple
from networks.CLmodel import ContinualLearningModelWrapper


def kmeanspp_with_fixed_anchors( embeddings: torch.Tensor, anchor_embeddings: torch.Tensor, anchor_indices: List[int], remaining_clusters: int,
    lloyd_max_iter: int = 300, n_init: int = 5, tol: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constrained K-Means with fixed anchor centers.

    Anchors:
      - are always selected
      - never move
      - participate in assignment

    Remaining centers:
      - initialized via K-Means++
      - updated with Lloyd iterations

    Returns:
        (labels_for_free_embeddings, all_centers)
    """

    device = embeddings.device
    n_samples, dim = embeddings.shape
    n_anchors = anchor_embeddings.size(0)

    # Mask anchors out of data
    free_mask = torch.ones(n_samples, dtype=torch.bool, device=device)
    free_mask[anchor_indices] = False
    free_embeddings = embeddings[free_mask]

    best_centers = None
    best_inertia = float("inf")
    best_labels = None

    for _ in tqdm(range(n_init), desc="Seeding iteration ... "):
        # -----------------------------
        # K-Means++ seeding (anchored)
        # -----------------------------
        centers = anchor_embeddings.clone()

        for _ in range(remaining_clusters):
            dists = torch.cdist(free_embeddings, centers).square()
            min_dists = dists.min(dim=1).values
            probs = min_dists / min_dists.sum()
            idx = torch.multinomial(probs, 1).item()
            centers = torch.cat(
                [centers, free_embeddings[idx].unsqueeze(0)], dim=0
            )

        # Indices of movable centers
        movable_slice = slice(n_anchors, centers.size(0))

        # -----------------------------
        # Lloyd iterations
        # -----------------------------
        for _ in range(lloyd_max_iter):
            # Assignment against ALL centers
            dists = torch.cdist(free_embeddings, centers)
            labels = dists.argmin(dim=1)

            new_centers = centers.clone()
            counts = torch.zeros(centers.size(0), device=device)

            # Accumulate only for movable centers
            for j in range(n_anchors, centers.size(0)):
                mask = labels == j
                if mask.any():
                    new_centers[j] = free_embeddings[mask].mean(dim=0)
                    counts[j] = mask.sum()

            # Handle empty movable clusters
            empty = (counts[n_anchors:] == 0).nonzero(as_tuple=True)[0]
            if len(empty) > 0:
                min_dists = dists.min(dim=1).values
                for e in empty:
                    farthest = min_dists.argmax()
                    new_centers[n_anchors + e] = free_embeddings[farthest]
                    min_dists[farthest] = 0.0

            # Convergence check (movable centers only)
            shift = (
                (new_centers[movable_slice] - centers[movable_slice])
                .norm(dim=1)
                .max()
                .item()
            )
            centers = new_centers

            if shift < tol:
                break

        # -----------------------------
        # Inertia (constrained objective)
        # -----------------------------
        dists = torch.cdist(free_embeddings, centers)
        labels = dists.argmin(dim=1)
        inertia = dists.gather(1, labels.unsqueeze(1)).square().sum().item()

        # Complete embeddings calculation
        dists = torch.cdist(embeddings, centers)
        labels = dists.argmin(dim=1)

        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers
            best_labels = labels

    return best_labels, best_centers


def kmeanspp(embeddings: torch.Tensor, n_clusters: int,
             lloyd_max_iter: int = 300, n_init: int = 5, tol: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """
    K-Means clustering entirely in PyTorch, staying on the input device.
    Uses K-Means++ initialization and runs `n_init` independent restarts,
    returning the result with the lowest inertia.

    Returns: (cluster_labels, cluster_centers)
    """
    n_samples, dim = embeddings.shape
    device = embeddings.device

    best_labels = None
    best_centers = None
    best_inertia = float('inf')

    for _ in tqdm(range(n_init), desc="Seeding variation ... "):
        # --- K-Means++ initialization ---
        center_indices = []
        idx = torch.randint(n_samples, (1,)).item()
        center_indices.append(idx)
        centers = embeddings[idx].unsqueeze(0)  # (1, dim)

        for _ in range(1, n_clusters):
            # Squared distances to nearest existing center
            dists = torch.cdist(embeddings, centers).square()  # (n, c)
            min_dists = dists.min(dim=1).values                # (n,)
            # Sample proportionally to squared distance
            probs = min_dists / min_dists.sum()
            idx = torch.multinomial(probs, 1).item()
            center_indices.append(idx)
            centers = torch.cat([centers, embeddings[idx].unsqueeze(0)], dim=0)

        # --- Lloyd iterations ---
        for _ in range(lloyd_max_iter):
            dists = torch.cdist(embeddings, centers)  # (n, k)
            labels = dists.argmin(dim=1)               # (n,)

            new_centers = torch.zeros_like(centers)
            counts = torch.zeros(n_clusters, device=device)
            new_centers.scatter_add_(0, labels.unsqueeze(1).expand(-1, dim), embeddings)
            counts.scatter_add_(0, labels, torch.ones(n_samples, device=device))

            # Handle empty clusters by re-seeding from the point farthest from any center
            empty_mask = counts == 0
            if empty_mask.any():
                min_dists_to_centers = dists.min(dim=1).values
                for empty_idx in empty_mask.nonzero(as_tuple=True)[0]:
                    farthest = min_dists_to_centers.argmax()
                    new_centers[empty_idx] = embeddings[farthest]
                    counts[empty_idx] = 1
                    min_dists_to_centers[farthest] = 0.0

            new_centers /= counts.unsqueeze(1)

            shift = (new_centers - centers).norm(dim=1).max().item()
            centers = new_centers
            if shift < tol:
                break
        # Compute final labels and inertia
        dists = torch.cdist(embeddings, centers)
        labels = dists.argmin(dim=1)
        inertia = dists.gather(1, labels.unsqueeze(1)).square().sum().item()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centers = centers

    return best_labels, best_centers


def knn_typicality(
    cluster_embeddings: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Compute typicality scores for a set of embeddings.
    Typicality = 1 / mean_distance_to_k_nearest_neighbors.
    """
    # Pairwise distances within the cluster
    dists = torch.cdist(cluster_embeddings, cluster_embeddings)  # (m, m)
    # Top-(k+1) smallest distances (includes self at distance ~0)
    topk_dists, _ = dists.topk(k + 1, dim=1, largest=False)
    # Exclude self (column 0) and compute mean over k neighbors
    mean_dist = topk_dists[:, 1:].mean(dim=1)
    typicality = 1.0 / (mean_dist + 1e-8)
    return typicality


def extract_embeddings(network: ContinualLearningModelWrapper, dataloader: DataLoader, torch_device: torch.device) -> torch.Tensor:
    """Extract embeddings from all samples, returned as a single tensor on `device`."""
    network.eval()
    network.to(torch_device)
    with torch.inference_mode():
        all_embeddings = []
        for input_model, _ in tqdm(dataloader, desc="Extracting embeddings"):
            input_model = input_model.to(torch_device)
            embeddings = network.get_embedding(input_model)
            all_embeddings.append(embeddings)

        embeddings = torch.cat(all_embeddings, dim=0)
        return embeddings
