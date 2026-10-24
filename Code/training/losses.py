from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LossOutput:
    total: torch.Tensor
    prediction: torch.Tensor
    temporal_perturbation: torch.Tensor
    relational_perturbation: torch.Tensor


class DualPerturbationLoss(nn.Module):

    def __init__(self, lambda_temporal: float, lambda_relational: float) -> None:
        super().__init__()
        if lambda_temporal < 0:
            raise ValueError("lambda_temporal must be non-negative")
        if lambda_relational < 0:
            raise ValueError("lambda_relational must be non-negative")
        self.lambda_temporal = lambda_temporal
        self.lambda_relational = lambda_relational
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> LossOutput:
        required = {
            "prediction_logits",
            "temporal_perturbation_logits",
            "relational_perturbation_logits",
        }
        missing = required.difference(outputs)
        if missing:
            raise ValueError(f"Model output is missing loss tensors: {sorted(missing)}")
        prediction = self.criterion(outputs["prediction_logits"], target)
        temporal = self.criterion(outputs["temporal_perturbation_logits"], target)
        relational = self.criterion(outputs["relational_perturbation_logits"], target)
        total = (
            prediction
            + self.lambda_temporal * temporal
            + self.lambda_relational * relational
        )
        return LossOutput(total, prediction, temporal, relational)


# Backward-compatible API used by the original release.
class CausalInterventionLoss(DualPerturbationLoss):

    def __init__(self, lambda_temporal: float, lambda_spatial: float) -> None:
        super().__init__(
            lambda_temporal=lambda_temporal,
            lambda_relational=lambda_spatial,
        )
