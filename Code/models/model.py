from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalStock(nn.Module):

    def __init__(
        self,
        time_length: int,
        feature_dim: int,
        hidden_dim: int = 32,
        classes: int = 2,
        dropout: float = 0.3,
        patch_size: int = 5,
        transformer_layers: int = 1,
        num_heads: int = 4,
        relation_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be greater than 0")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if relation_temperature <= 0:
            raise ValueError("relation_temperature must be greater than 0")

        self.time_length = time_length
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.num_patches = math.ceil(time_length / patch_size)
        self.relation_temperature = relation_temperature

        self.patch_embedding = nn.Linear(patch_size, hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.num_patches, hidden_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.stock_mlp = nn.Sequential(
            nn.Linear(feature_dim * hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.mask_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.raw_tau = nn.Parameter(torch.tensor(0.0))
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, classes),
        )
        nn.init.normal_(self.position_embedding, std=0.02)

    def _patch_encode(self, x: torch.Tensor) -> torch.Tensor:
        series = x.transpose(1, 2)
        padded_length = self.num_patches * self.patch_size
        if padded_length > self.time_length:
            series = F.pad(series, (0, padded_length - self.time_length))
        patches = series.unfold(-1, self.patch_size, self.patch_size)
        stocks, indicators, patches_count, _ = patches.shape
        patches = patches.reshape(stocks * indicators, patches_count, self.patch_size)
        encoded = self.patch_embedding(patches) + self.position_embedding
        encoded = self.temporal_encoder(encoded).mean(dim=1)
        encoded = encoded.reshape(stocks, indicators * self.hidden_dim)
        return self.stock_mlp(encoded)

    def _straight_through_mask(
        self, scores: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        hard = (scores > tau).to(scores.dtype)
        soft = torch.sigmoid((scores - tau) / self.relation_temperature)
        return hard.detach() - soft.detach() + soft

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Input must have shape N×T×D; received {tuple(x.shape)}")
        if tuple(x.shape[1:]) != (self.time_length, self.feature_dim):
            raise ValueError(
                f"Expected the final two dimensions to be "
                f"{(self.time_length, self.feature_dim)}; received {tuple(x.shape[1:])}"
            )

        # Eq. (5)-(7): patch-wise stock temporal representation H.
        temporal = self._patch_encode(x)

        # Eq. (8): complementary temporal masks and H_P/H_F.
        mask_score = self.mask_mlp(temporal)
        predictive_mask = torch.sigmoid(mask_score)
        complementary_mask = torch.sigmoid(-mask_score)
        h_p = predictive_mask * temporal
        h_f = complementary_mask * temporal

        # Eq. (9)-(10): relation scores, hard STE mask, A_P and A_F.
        relation_scores = torch.sigmoid(
            (self.query(h_p) @ self.key(h_p).transpose(0, 1))
            / math.sqrt(self.hidden_dim)
        )
        tau = torch.sigmoid(self.raw_tau)
        relation_mask = self._straight_through_mask(relation_scores, tau)
        a_p = relation_mask * relation_scores
        a_f = (1.0 - relation_mask) * (1.0 - relation_scores)

        # Eq. (11): predictive spatio-temporal representation Z_P.
        value_p = self.value(h_p)
        z_p = torch.cat((h_p, a_p @ value_p), dim=-1)

        # Eq. (12): temporal and relational perturbation representations.
        h_temporal_perturb = h_p + h_f
        z_t_pt = torch.cat(
            (
                h_temporal_perturb,
                a_p @ self.value(h_temporal_perturb),
            ),
            dim=-1,
        )
        z_r_pt = torch.cat((h_p, (a_p + a_f) @ value_p), dim=-1)

        prediction_logits = self.predictor(z_p)
        temporal_perturbation_logits = self.predictor(z_t_pt)
        relational_perturbation_logits = self.predictor(z_r_pt)

        return {
            # Paper-aligned names.
            "prediction_logits": prediction_logits,
            "temporal_perturbation_logits": temporal_perturbation_logits,
            "relational_perturbation_logits": relational_perturbation_logits,
            "predictive_adjacency": a_p,
            "perturbation_adjacency": a_f,
            "predictive_mask": predictive_mask,
            "complementary_mask": complementary_mask,
            "relation_scores": relation_scores,
            "relation_mask": relation_mask,
            "tau": tau,
            # Backward-compatible aliases used by earlier code.
            "logits": prediction_logits,
            "temporal_do_logits": temporal_perturbation_logits,
            "spatial_do_logits": relational_perturbation_logits,
            "a_c": a_p,
            "a_f": a_f,
        }


# Backward-compatible import name used by the original release.
DualPerturb = CausalStock
