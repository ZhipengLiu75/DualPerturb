from __future__ import annotations

import math

import torch
from torch import nn


def extract_logits(outputs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(outputs, dict):
        if "prediction_logits" in outputs:
            return outputs["prediction_logits"]
        if "logits" in outputs:
            return outputs["logits"]
        raise ValueError("Model output dictionary does not contain prediction logits")
    return outputs


def matthews_correlation(tp: int, tn: int, fp: int, fn: int) -> float:
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator == 0:
        return 0.0
    return (tp * tn - fp * fn) / math.sqrt(denominator)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
    days: range,
    time_length: int,
) -> tuple[float, float, float]:
    if len(days) == 0:
        raise ValueError("Evaluation days must not be empty")
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    total = 0
    tp = tn = fp = fn = 0
    for day in days:
        x = data[:, day - time_length : day, :]
        target = labels[:, day]
        logits = extract_logits(model(x))
        loss_sum += criterion(logits, target).item()
        prediction = logits.argmax(dim=1)
        correct += (prediction == target).sum().item()
        total += target.numel()
        tp += ((prediction == 1) & (target == 1)).sum().item()
        tn += ((prediction == 0) & (target == 0)).sum().item()
        fp += ((prediction == 1) & (target == 0)).sum().item()
        fn += ((prediction == 0) & (target == 1)).sum().item()
    return (
        loss_sum / len(days),
        correct / total,
        matthews_correlation(tp, tn, fp, fn),
    )
