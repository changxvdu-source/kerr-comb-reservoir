"""Score a ring saved by train_physical.py on the held-out test series.

Also re-runs it with half the time step and twice the spectral grid, to check
that the result is a property of the device and not of the discretisation.

Example:
    python experiments/score.py --ring results/train_multimode.pt --horizon 10
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from combrc import load_ring, make_series, score_baselines, score_ring  # noqa: E402
from combrc.evaluate import N_SYMBOLS, WARMUP  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ring", required=True)
    ap.add_argument("--task", default="mackey_glass", choices=["mackey_glass", "rossler", "lorenz"])
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--n-lines", type=int, default=31)
    ap.add_argument("--series-seed", type=int, default=0, help="0 is the test series used by train_physical.py")
    ap.add_argument("--baselines", action="store_true")
    args = ap.parse_args()

    u = torch.tensor(make_series(args.task, WARMUP + N_SYMBOLS + args.horizon, seed=args.series_seed),
                     dtype=torch.float64)
    ring = load_ring(args.ring)
    fine = load_ring(args.ring, n_modes=2 * ring.n_modes, steps_per_symbol=2 * ring.steps_per_symbol)
    if args.baselines:
        print("baselines:", {k: f"{v:.2e}" for k, v in score_baselines(u, args.horizon).items()})
    print("as trained:  ", score_ring(ring, u, args.horizon, args.n_lines))
    print("finer grid:  ", score_ring(fine, u, args.horizon, args.n_lines))
    print("parameters:  ", {k: f"{v:.3g}" for k, v in ring.describe().items()})


if __name__ == "__main__":
    main()
