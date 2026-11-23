"""Stronger software baselines under the paper's protocol: larger ESNs and next-generation reservoir computing.

The encoding comparison measured the ring against an ESN of the ring's size (31 nodes). A reviewer will
ask how far that is from what software does at a modest size. Here, per task and with the same splits,
input noise and fresh-run protocol as compare_encodings.py:

  ESN 31 / 100 / 300   tanh echo state network, readout over the same number of delay taps as the ring;
                       spectral radius and input scale chosen on the validation part (mean over
                       ``--select-pairs`` input-noise realisations), test error over ``--pairs`` new ones;
                       ``--esn-seeds`` random reservoirs per size are part of the choice
  NG-RC                next-generation reservoir computing (Gauthier et al. 2021): a constant, the k most
                       inputs spaced s apart and all their unique quadratic products, ridge readout; k and s
                       chosen on the validation part the same way

    python experiments/strong_baselines.py
"""
import argparse
import csv
import itertools
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import compare_encodings as ce  # noqa: E402
from combrc import fit_readout, make_task, nmse  # noqa: E402
from combrc.evaluate import WARMUP, _splits, esn_states  # noqa: E402
from combrc.readout import add_taps  # noqa: E402
from combrc.tasks import is_symbol_task, symbol_error_rate  # noqa: E402

# task -> (readout taps of the ring's main variant, test symbols or 0 for the default split)
TASKS = {"mackey_glass_h10": (8, 0), "channel_eq": (8, 5000), "narma10": (12, 0), "fiber_pam4_17km": (8, 5000)}


def ngrc_features(u, cfg):
    """[T, 1 + k + k(k+1)/2]: constant, k delayed inputs u[t], u[t-s], ..., u[t-(k-1)s], and their unique
    pairwise products; cfg = (k, s)."""
    k, skip = cfg
    t = torch.arange(len(u))
    lags = torch.stack([u[(t - j * skip).clamp_min(0)] for j in range(k)], 1)
    quad = [lags[:, i] * lags[:, j] for i, j in itertools.combinations_with_replacement(range(k), 2)]
    return torch.cat([torch.ones(len(u), 1, dtype=u.dtype), lags, torch.stack(quad, 1)], 1)


RIDGE_DEVICE = "cpu"


def evaluate(make_features, configs, u, target, input_noise, n_train, n_val, select_pairs, pairs, error_rate):
    """Choose the config by the mean validation NMSE over ``select_pairs`` noise realisations, report the mean
    test NMSE (and error rate) over ``pairs`` new realisations."""
    def run(cfg, noise_seed):
        gen = torch.Generator().manual_seed(noise_seed)
        seen = u + input_noise * torch.randn(u.shape, generator=gen, dtype=u.dtype) if input_noise else u
        x = make_features(seen, cfg)[WARMUP:].to(RIDGE_DEVICE)
        t = target.to(RIDGE_DEVICE)
        pred = fit_readout(x, t, n_train, n_val).predict(x)
        test = slice(n_train + n_val, None)
        return (nmse(t[n_train:n_train + n_val], pred[n_train:n_train + n_val]).item(),
                nmse(t[test], pred[test]).item(),
                error_rate(target[test].numpy(), pred[test].cpu().numpy()) if error_rate else float("nan"))
    best = min(configs, key=lambda c: np.mean([run(c, k)[0] for k in range(select_pairs)]))
    scores = [run(best, 100 + k) for k in range(pairs)]
    return float(np.mean([s[1] for s in scores])), float(np.mean([s[2] for s in scores])), best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--noise", nargs="+", default=["realistic"], choices=["none", "realistic"])
    ap.add_argument("--series-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sizes", type=int, nargs="+", default=[31, 100, 300])
    ap.add_argument("--esn-seeds", type=int, default=3)
    ap.add_argument("--ngrc-k", type=int, nargs="+", default=[2, 4, 8, 12, 16])
    ap.add_argument("--ngrc-skip", type=int, nargs="+", default=[1, 2, 4, 6])
    ap.add_argument("--select-pairs", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--delay-sizes", type=int, nargs="*", default=[],
                    help="also ESNs of these sizes whose input carries u[t] and u[t-d] (the delay line of the ring's "
                         "mask_delay groups), d chosen on the validation part from 1..12")
    ap.add_argument("--fit-symbols", type=int, default=0,
                    help="training + validation length (default 1360); tasks without a test length then use 1000")
    ap.add_argument("--radii", type=float, nargs="+", default=[0.5, 0.8, 0.95, 1.1])
    ap.add_argument("--scales", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    ap.add_argument("--ridge-device", default="cpu", help="where the readouts are fitted ('cuda' for many features)")
    ap.add_argument("--input-noise", type=float, default=None,
                    help="rms input noise of the 'realistic' setting (default: the ring's source noise referred to "
                         "the input, compare_encodings.REALISTIC['baseline_input_noise'])")
    ap.add_argument("--out", default="results/strong_baselines")
    args = ap.parse_args()
    global RIDGE_DEVICE
    RIDGE_DEVICE = args.ridge_device

    rows = []
    for task in args.tasks:
        taps, n_test = TASKS.get(task, (8, 5000 if task.startswith(("channel_eq", "fiber")) else 0))
        if args.fit_symbols and not n_test:
            n_test = 1000
        ce.set_length(n_test, None, args.fit_symbols or None)
        n_train, n_val = _splits(ce.N_TOTAL, ce.N_TEST)
        error_rate = symbol_error_rate if is_symbol_task(task) else None
        for noise in args.noise:
            realistic = ce.REALISTIC["baseline_input_noise"] if args.input_noise is None else args.input_noise
            input_noise = realistic if noise == "realistic" else 0.0
            per = {}
            for seed in args.series_seeds:
                u, y = (torch.tensor(a, dtype=torch.float64) for a in make_task(task, WARMUP + ce.N_TOTAL, seed=seed))
                u, target = u[: WARMUP + ce.N_TOTAL], y[WARMUP: WARMUP + ce.N_TOTAL]
                for size in args.sizes:
                    grid = [(r, s, e) for r in args.radii for s in args.scales for e in range(args.esn_seeds)]

                    def esn(seen, cfg, size=size):
                        r, s, e = cfg
                        return add_taps(esn_states(seen, size, r, s, seed=e), taps)
                    per.setdefault(f"ESN {size}", []).append(
                        evaluate(esn, grid, u, target, input_noise, n_train, n_val, args.select_pairs, args.pairs,
                                 error_rate))
                for size in args.delay_sizes:
                    from esn_delay import esn2_states, shifted
                    grid = [(r, s_, 0, d) for r in (0.5, 0.8, 0.95, 1.1) for s_ in (0.1, 0.3, 1.0) for d in (1, 2, 3, 4, 6, 8, 9, 10, 12)]

                    def esn_d(seen, cfg, size=size):
                        r, s_, e, d = cfg
                        return add_taps(esn2_states(seen, shifted(seen, d), size, r, s_, seed=e), taps)
                    per.setdefault(f"ESN {size} + delay", []).append(
                        evaluate(esn_d, grid, u, target, input_noise, n_train, n_val, args.select_pairs, args.pairs,
                                 error_rate))
                per.setdefault("NG-RC", []).append(
                    evaluate(ngrc_features, [(k, s_) for k in args.ngrc_k for s_ in args.ngrc_skip], u, target, input_noise, n_train, n_val, args.select_pairs,
                             args.pairs, error_rate))
                print(task, noise, "series", seed, {k: f"{v[-1][0]:.2e}" for k, v in per.items()}, flush=True)
            for name, results in per.items():
                row = {"task": task, "noise": noise, "model": name,
                       "test_nmse": ce.gmean([r[0] for r in results]),
                       "ser": float(np.mean([r[1] for r in results])) if error_rate else float("nan"),
                       "chosen": " | ".join(str(r[2]) for r in results)}
                rows.append(row)
                print(f"  {task:18s} {noise:9s} {name:8s} NMSE {row['test_nmse']:.3e}"
                      + (f"  SER {row['ser']:.2e}" if error_rate else ""), flush=True)
    with open(args.out + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("saved", args.out + ".csv")


if __name__ == "__main__":
    main()
