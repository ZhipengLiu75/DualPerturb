from __future__ import annotations

import torch
from torch import nn

from evaluation import evaluate


def test_model(
    model: nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
    test_days: range,
    time_length: int,
) -> tuple[float, float, float]:
    return evaluate(model, data, labels, test_days, time_length)
