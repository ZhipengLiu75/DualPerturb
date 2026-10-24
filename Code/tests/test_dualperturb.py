from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from data_processing import load_data, prepare_data, time_split
from evaluation import matthews_correlation
from models.model import DualPerturb
from training.losses import DualPerturbationLoss


class DualPerturbTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_equations_shapes_and_backward(self) -> None:
        torch.manual_seed(7)
        model = DualPerturb(
            time_length=21,
            feature_dim=5,
            hidden_dim=16,
            dropout=0.0,
            patch_size=5,
            num_heads=4,
            relation_temperature=0.1,
        )
        x = torch.randn(7, 21, 5)
        target = torch.randint(0, 2, (7,))
        outputs = model(x)

        self.assertEqual(tuple(outputs["prediction_logits"].shape), (7, 2))
        self.assertEqual(
            tuple(outputs["temporal_perturbation_logits"].shape), (7, 2)
        )
        self.assertEqual(
            tuple(outputs["relational_perturbation_logits"].shape), (7, 2)
        )
        self.assertEqual(tuple(outputs["predictive_adjacency"].shape), (7, 7))
        self.assertEqual(tuple(outputs["perturbation_adjacency"].shape), (7, 7))

        self.assertTrue(
            torch.allclose(
                outputs["predictive_mask"] + outputs["complementary_mask"],
                torch.ones_like(outputs["predictive_mask"]),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs["predictive_adjacency"],
                outputs["relation_mask"] * outputs["relation_scores"],
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs["perturbation_adjacency"],
                (1.0 - outputs["relation_mask"])
                * (1.0 - outputs["relation_scores"]),
            )
        )
        self.assertTrue(
            torch.all(
                (outputs["relation_mask"] == 0)
                | (outputs["relation_mask"] == 1)
            )
        )

        loss = DualPerturbationLoss(0.5, 0.5)(outputs, target)
        self.assertTrue(torch.isfinite(loss.total))
        loss.total.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, msg=f"missing gradient: {name}")
            self.assertTrue(
                torch.isfinite(parameter.grad).all(),
                msg=f"non-finite gradient: {name}",
            )
        self.assertIsNotNone(model.raw_tau.grad)

    def test_real_data_step_and_splits(self) -> None:
        raw = load_data("CSI300")
        normalized, labels = prepare_data(raw)
        train_days, valid_days, test_days = time_split(raw.shape[1], 20)
        self.assertEqual((len(train_days), len(valid_days), len(test_days)), (1169, 255, 255))

        model = DualPerturb(
            time_length=20,
            feature_dim=raw.shape[2],
            hidden_dim=16,
            dropout=0.0,
            patch_size=5,
            num_heads=4,
        )
        x = torch.from_numpy(normalized[:, :20, :])
        target = torch.from_numpy(labels[:, 20])
        outputs = model(x)
        loss = DualPerturbationLoss(0.5, 0.5)(outputs, target)
        loss.total.backward()
        self.assertTrue(torch.isfinite(loss.total))

    def test_matthews_correlation(self) -> None:
        self.assertEqual(matthews_correlation(5, 7, 0, 0), 1.0)
        self.assertEqual(matthews_correlation(0, 0, 5, 7), -1.0)
        self.assertEqual(matthews_correlation(0, 10, 0, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
