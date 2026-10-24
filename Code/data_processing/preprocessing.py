from __future__ import annotations

import numpy as np


def normalize_features(data: np.ndarray) -> np.ndarray:
    train_end = int(data.shape[1] * 0.70)
    train_data = data[:, :train_end, :]
    mean = train_data.mean(axis=1, keepdims=True)
    std = train_data.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    return ((data - mean) / std).astype(np.float32)


def create_direction_labels(data: np.ndarray) -> np.ndarray:
    close = data[:, :, 1]
    returns = np.zeros_like(close, dtype=np.float32)
    np.divide(
        close[:, 1:] - close[:, :-1],
        close[:, :-1],
        out=returns[:, 1:],
        where=close[:, :-1] != 0,
    )
    return (returns > 0).astype(np.int64)


def prepare_data(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return normalize_features(data), create_direction_labels(data)
