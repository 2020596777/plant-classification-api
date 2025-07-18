import torch
import numpy as np

# from typing import Tuple

def _to_uint8(images: torch.Tensor) -> np.ndarray:
    if images.dtype not in (torch.float32, torch.float64):
        raise TypeError("Images must be float tensors in [0, 1] range.")

    #Detach from graph, ensure on CPU, convert to uint8.
    return (images.detach().cpu().numpy() * 255).astype(np.uint8)


def extract_color_histogram(
    images: torch.Tensor,
    bins: int = 32,
) -> torch.Tensor:

    # Convert to (B,3,H,W) uint8 NumPy array
    imgs_np: np.ndarray = _to_uint8(images)
    B, _, _, _ = imgs_np.shape

    # Pre‑allocate output array: (B,3*bins)
    hist_features = np.zeros((B, 3 * bins), dtype=np.float32)

    # Compute histogram per image & per channel
    for i in range(B):
        offset = 0
        for c in range(3):
            channel_flat = imgs_np[i, c].ravel()
            h, _ = np.histogram(channel_flat, bins=bins, range=(0, 255))
            h = h.astype(np.float32)
            h /= h.sum()  #normalise so each channel sums to 1
            hist_features[i, offset : offset + bins] = h
            offset += bins

    # Wrap into tensor on the same device as input
    return torch.from_numpy(hist_features).to(images.device)


def extract_texture_features(images: torch.Tensor) -> torch.Tensor:

    imgs_np: np.ndarray = images.detach().cpu().numpy()  # (B,3,H,W)
    gray_np = imgs_np.mean(axis=1)  # -> (B,H,W)

    mean_vals = gray_np.mean(axis=(1, 2))  # (B,)
    var_vals = gray_np.var(axis=(1, 2))    # (B,)

    #Stack as (B,10) by repeating along last dimension
    feats_np = np.stack([mean_vals, var_vals], axis=1)  # (B,2)
    feats_np = np.repeat(feats_np, repeats=5, axis=1)   # (B,10)

    return torch.from_numpy(feats_np.astype(np.float32)).to(images.device)
