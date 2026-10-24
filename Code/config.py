from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="DualPerturb")
    parser.add_argument("--market", default="CSI300")
    parser.add_argument("--time-length", type=int, default=20)
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--relation-temperature", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lambda-temporal", type=float, default=0.5)
    parser.add_argument(
        "--lambda-relational",
        "--lambda-spatial",
        dest="lambda_relational",
        type=float,
        default=0.5,
    )
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be greater than 0")
    if args.lr_patience < 0:
        parser.error("--lr-patience must be non-negative")
    if args.early_stop_patience <= 0:
        parser.error("--early-stop-patience must be greater than 0")
    return args
