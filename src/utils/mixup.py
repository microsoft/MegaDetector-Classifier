"""Mixup utilities ported from SINCHI_DINOV3.

Source: /shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/utils/mixup.py
Commit SHA: no git provenance (working tree copy)
RNG source: numpy.random.beta (for lambda) and torch.randint (for mixup
             partner selection). Tests monkeypatch these for determinism.
"""

import numpy as np
import torch
from typing import Tuple

__all__ = ["noisy_same_class_mixup", "efficient_batch_same_class_mixup"]


def noisy_same_class_mixup(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    mixup_alpha: float = 0.2,
    noise_std: float = 0.01,
    noise_type: str = "gaussian",
    dropout_prob: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply noisy mixup to embeddings from the same class.
    
    For each embedding, finds another embedding from the same class,
    mixes them, and adds noise. This creates intra-class augmentation
    while preserving class identity.
    
    Args:
        embeddings: Tensor of shape (B, D) where D is embedding dimension
        labels: Tensor of shape (B,) with class indices
        mixup_alpha: Beta distribution parameter for mixup lambda.
            Higher values (0.4-1.0) = more aggressive mixing.
            Lower values (0.1-0.2) = conservative mixing.
        noise_std: Standard deviation for Gaussian noise.
            Should be small relative to embedding scale (typically 0.01-0.05).
        noise_type: Type of noise to apply.
            Options: "gaussian", "dropout", "both"
        dropout_prob: Dropout probability if using dropout noise.
            Only used if noise_type is "dropout" or "both".
    
    Returns:
        Tuple of (mixed_noisy_embeddings, labels)
        Labels remain the same since both samples are from same class.
    """
    batch_size = embeddings.size(0)
    device = embeddings.device
    
    # Create index mapping: for each sample, find another sample from same class
    same_class_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for i in range(batch_size):
        current_label = labels[i]
        # Find all indices with the same label
        same_class_mask = (labels == current_label)
        same_class_idxs = torch.where(same_class_mask)[0]
        
        # If only one sample of this class, use itself (no mixup)
        if len(same_class_idxs) == 1:
            same_class_indices[i] = i
        else:
            # Randomly select another sample from same class (excluding current)
            candidates = same_class_idxs[same_class_idxs != i]
            if len(candidates) > 0:
                rand_idx = torch.randint(0, len(candidates), (1,), device=device).item()
                same_class_indices[i] = candidates[rand_idx]
            else:
                same_class_indices[i] = i
    
    # Sample mixup lambda from Beta distribution
    if mixup_alpha > 0:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        # Clamp to [0.1, 0.9] to ensure meaningful mixing
        lam = max(0.1, min(0.9, lam))
    else:
        lam = 0.5
    
    # Mix embeddings: λ * emb1 + (1-λ) * emb2
    paired_embeddings = embeddings[same_class_indices]
    mixed_embeddings = lam * embeddings + (1 - lam) * paired_embeddings
    
    # Add noise
    if noise_type == "gaussian":
        noise = torch.randn_like(mixed_embeddings) * noise_std
        noisy_embeddings = mixed_embeddings + noise
    elif noise_type == "dropout":
        # Apply dropout-like noise (randomly zero some features)
        dropout_mask = torch.rand_like(mixed_embeddings) > dropout_prob
        noisy_embeddings = mixed_embeddings * dropout_mask.float()
    elif noise_type == "both":
        # Combine both: apply dropout, then add Gaussian noise
        dropout_mask = torch.rand_like(mixed_embeddings) > dropout_prob
        gaussian_noise = torch.randn_like(mixed_embeddings) * noise_std
        noisy_embeddings = (mixed_embeddings * dropout_mask.float()) + gaussian_noise
    else:
        # No noise, just mixup
        noisy_embeddings = mixed_embeddings
    
    # Labels remain the same (both samples from same class)
    return noisy_embeddings, labels


def efficient_batch_same_class_mixup(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    mixup_alpha: float = 0.2,
    noise_std: float = 0.01,
    noise_type: str = "gaussian",
    dropout_prob: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    More efficient batch-level implementation using vectorized operations.
    
    This version uses a permutation-based approach that's faster for large batches,
    but may occasionally pair samples from different classes (which we filter out).
    
    Args:
        embeddings: Tensor of shape (B, D) where D is embedding dimension
        labels: Tensor of shape (B,) with class indices
        mixup_alpha: Beta distribution parameter for mixup lambda
        noise_std: Standard deviation for Gaussian noise
        noise_type: Type of noise ("gaussian", "dropout", "both")
        dropout_prob: Dropout probability if using dropout noise
    
    Returns:
        Tuple of (mixed_noisy_embeddings, labels)
    """
    batch_size = embeddings.size(0)
    device = embeddings.device
    dtype = embeddings.dtype
    
    # Create permutation indices
    indices = torch.randperm(batch_size, device=device)
    
    # Find pairs within same class
    # Strategy: shuffle and pair, but only keep pairs where labels match
    shuffled_labels = labels[indices]
    same_class_mask = (labels == shuffled_labels)
    
    # For samples where we can pair with same class, use shuffled index
    # Otherwise, use original index (no mixup for that sample)
    mixup_indices = torch.where(
        same_class_mask,
        indices,
        torch.arange(batch_size, device=device)
    )
    
    # Sample lambda for each sample (but we'll use a single lambda for consistency)
    if mixup_alpha > 0:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        lam = max(0.1, min(0.9, lam))
    else:
        lam = 0.5
    
    # Mix embeddings
    paired_embeddings = embeddings[mixup_indices]
    mixed_embeddings = lam * embeddings + (1 - lam) * paired_embeddings
    
    # Add noise
    if noise_type == "gaussian":
        noise = torch.randn_like(mixed_embeddings) * noise_std
        noisy_embeddings = mixed_embeddings + noise
    elif noise_type == "dropout":
        dropout_mask = torch.rand_like(mixed_embeddings) > dropout_prob
        noisy_embeddings = mixed_embeddings * dropout_mask.float()
    elif noise_type == "both":
        dropout_mask = torch.rand_like(mixed_embeddings) > dropout_prob
        gaussian_noise = torch.randn_like(mixed_embeddings) * noise_std
        noisy_embeddings = (mixed_embeddings * dropout_mask.float()) + gaussian_noise
    else:
        noisy_embeddings = mixed_embeddings
    
    return noisy_embeddings, labels
