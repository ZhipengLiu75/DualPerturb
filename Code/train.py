from __future__ import annotations

import copy

import numpy as np

from config import parse_args
from training import train_model
from utils import set_seed


def main() -> None:
    args = parse_args()
    results = []
    for run_index in range(args.runs):
        run_args = copy.deepcopy(args)
        run_args.seed = args.seed + run_index
        print(f"run={run_index + 1}/{args.runs}, seed={run_args.seed}")
        set_seed(run_args.seed)
        results.append(train_model(run_args))

    acc = np.asarray([result.test_acc for result in results], dtype=np.float64)
    mcc = np.asarray([result.test_mcc for result in results], dtype=np.float64)
    ddof = 1 if args.runs > 1 else 0
    print(
        f"summary over {args.runs} run(s): "
        f"ACC (%)={100.0 * acc.mean():.2f} +/- {100.0 * acc.std(ddof=ddof):.2f}, "
        f"MCC={mcc.mean():.4f} +/- {mcc.std(ddof=ddof):.4f}"
    )


if __name__ == "__main__":
    main()
