from __future__ import annotations

from torch import nn

from models.model import CausalStock, DualPerturb


MODELS: dict[str, type[nn.Module]] = {
    "DualPerturb": DualPerturb,
    "CausalStock": CausalStock,
}


def build_model(args, feature_dim: int) -> nn.Module:
    if args.model not in MODELS:
        raise ValueError(f"Unknown model {args.model}; available models: {sorted(MODELS)}")
    return MODELS[args.model](
        time_length=args.time_length,
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        classes=2,
        dropout=args.dropout,
        patch_size=args.patch_size,
        transformer_layers=args.transformer_layers,
        num_heads=args.num_heads,
        relation_temperature=args.relation_temperature,
    )
