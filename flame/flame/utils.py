"""Utility helpers for compression and randomness."""
from __future__ import annotations

from typing import Iterable, List
import numpy as np


def quantize_dequantize(arr: np.ndarray, num_bits: int = 8) -> np.ndarray:
    """Fake-quantize to num_bits and back to float32 to simulate quantization effects."""
    if arr.size == 0:
        return arr
    a = arr.astype(np.float32)
    min_v = a.min()
    max_v = a.max()
    if max_v == min_v:
        return a
    levels = (1 << num_bits) - 1
    scaled = (a - min_v) / (max_v - min_v)
    q = np.round(scaled * levels) / levels
    deq = q * (max_v - min_v) + min_v
    return deq.astype(np.float32)


def sparsify(arr: np.ndarray, sparsity: float) -> np.ndarray:
    """Zero out smallest-magnitude values to reach the given sparsity in [0,1)."""
    if arr.size == 0 or sparsity <= 0:
        return arr
    k = int(arr.size * sparsity)
    if k <= 0:
        return arr
    flat = arr.ravel()
    thresh = np.partition(np.abs(flat), k - 1)[k - 1]
    mask = np.abs(flat) >= thresh
    out = np.zeros_like(flat)
    out[mask] = flat[mask]
    return out.reshape(arr.shape)


def add_gaussian_noise(arr: np.ndarray, std: float) -> np.ndarray:
    if std <= 0:
        return arr
    return arr + np.random.normal(0.0, std, size=arr.shape).astype(arr.dtype)
