from __future__ import annotations

from pathlib import Path
import re

import numpy as np


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_data(market: str) -> np.ndarray:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", market):
        raise ValueError("market may contain only letters, numbers, underscores, and hyphens")
    path = DATA_DIR / f"{market}.npy"
    if not path.exists():
        available = sorted(item.stem for item in DATA_DIR.glob("*.npy"))
        raise FileNotFoundError(f"Data file not found: {path}; available datasets: {available}")
    data = np.load(path).astype(np.float32)
    if data.ndim != 3:
        raise ValueError(f"Data must have shape N×T×D; received {data.shape}")
    if data.shape[2] < 2:
        raise ValueError("D must be at least 2; the second feature is assumed to be the closing price")
    if not np.isfinite(data).all():
        raise ValueError("Data contains NaN or infinite values")
    return data
