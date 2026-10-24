from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from data_processing import load_data, prepare_data, time_split
from evaluation import evaluate
from models import build_model
from testing import test_model
from training.losses import DualPerturbationLoss
from utils import resolve_device


@dataclass
class TrainingResult:
    model: nn.Module
    test_loss: float
    test_acc: float
    test_mcc: float


def train_model(args) -> TrainingResult:
    raw = load_data(args.market)
    normalized, labels_np = prepare_data(raw)
    train_days, valid_days, test_days = time_split(raw.shape[1], args.time_length)
    device = resolve_device(args.device)
    print(f"market={args.market}, shape={raw.shape}, device={device}")
    print(
        "time split: "
        f"train={len(train_days)}, valid={len(valid_days)}, "
        f"test={len(test_days)} usable prediction days (70%/15%/15%)"
    )
    data = torch.from_numpy(normalized).to(device)
    labels = torch.from_numpy(labels_np).to(device)
    model = build_model(args, raw.shape[2]).to(device)
    criterion = DualPerturbationLoss(
        lambda_temporal=args.lambda_temporal,
        lambda_relational=args.lambda_relational,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )
    best_state = copy.deepcopy(model.state_dict())
    best_valid_loss = float("inf")
    early_stop_count = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled_days = list(train_days)
        random.shuffle(shuffled_days)
        train_loss = 0.0
        prediction_loss = 0.0
        temporal_loss = 0.0
        relational_loss = 0.0
        predictive_mask_mean = 0.0
        relation_density = 0.0
        tau_mean = 0.0
        for day in shuffled_days:
            x = data[:, day - args.time_length : day, :]
            target = labels[:, day]
            optimizer.zero_grad()
            outputs = model(x)
            losses = criterion(outputs, target)
            losses.total.backward()
            optimizer.step()
            train_loss += losses.total.item()
            prediction_loss += losses.prediction.item()
            temporal_loss += losses.temporal_perturbation.item()
            relational_loss += losses.relational_perturbation.item()
            predictive_mask_mean += outputs["predictive_mask"].detach().mean().item()
            relation_density += outputs["relation_mask"].detach().mean().item()
            tau_mean += outputs["tau"].detach().item()
        valid_loss, valid_acc, valid_mcc = evaluate(
            model, data, labels, valid_days, args.time_length
        )
        scheduler.step(valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            early_stop_count = 0
        else:
            early_stop_count += 1
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss / len(shuffled_days):.4f} "
            f"prediction_loss={prediction_loss / len(shuffled_days):.4f} "
            f"temporal_loss={temporal_loss / len(shuffled_days):.4f} "
            f"relational_loss={relational_loss / len(shuffled_days):.4f} "
            f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f} "
            f"valid_mcc={valid_mcc:.4f} "
            f"predictive_mask_mean={predictive_mask_mean / len(shuffled_days):.4f} "
            f"relation_density={relation_density / len(shuffled_days):.4f} "
            f"tau={tau_mean / len(shuffled_days):.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"early_stop={early_stop_count}/{args.early_stop_patience}"
        )
        if early_stop_count >= args.early_stop_patience:
            print(f"early stopping at epoch {epoch}")
            break
    model.load_state_dict(best_state)
    test_loss, test_acc, test_mcc = test_model(
        model, data, labels, test_days, args.time_length
    )
    print(
        f"test_loss={test_loss:.4f} "
        f"test_acc={test_acc:.4f} test_mcc={test_mcc:.4f}"
    )
    return TrainingResult(model, test_loss, test_acc, test_mcc)
