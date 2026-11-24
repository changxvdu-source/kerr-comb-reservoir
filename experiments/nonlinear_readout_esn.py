"""Software side of nonlinear_readout.py: the ESN with the same linear and quadratic readouts over a sweep of
input noise (the paper's calibration variable) and two sizes. No ring simulation; same series, splits and
fitting protocol as nonlinear_readout.py.

    python experiments/nonlinear_readout_esn.py
"""
import argparse
import csv
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import compare_encodings as ce  # noqa: E402
import nonlinear_readout as nr  # noqa: E402
from combrc import make_task  # noqa: E402
from combrc.evaluate import WARMUP  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--noises", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3])
    ap.add_argument("--sizes", type=int, nargs="+", default=[31, 100])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default="results/nonlinear_readout_esn")
    args = ap.parse_args()
    torch.set_num_threads(4)
    n = ce.N_TOTAL
    n_train, n_val = int(0.6 * n), int(0.2 * n)
    sl = slice(n_train + n_val, n)
    rows = []
    for size in args.sizes:
        for noise in args.noises:
            res = {k: [] for k in ("linear K=1", "quadratic K=1", "linear K=4", "quadratic K=4")}
            for seed in args.series_seeds:
                u_np, y_np = make_task("mackey_glass_h10", WARMUP + n, seed=seed)
                u = torch.tensor(u_np, dtype=torch.float64)[: WARMUP + n]
                y = torch.tensor(y_np, dtype=torch.float64)[WARMUP: WARMUP + n]
                feats = nr.esn_runs(u, y, args.runs, noise, n_train, n_val, size=size)
                q = (lambda x: nr.quad(x)) if size == nr.LINES else (lambda x: quad_n(x, size))
                tx = feats[-1]
                for k in (1, 4):
                    res[f"linear K={k}"].append(nr.ridge(feats[:k], [y] * k, n_train, n_val, tx[sl], y[sl]))
                    res[f"quadratic K={k}"].append(nr.ridge([q(x) for x in feats[:k]], [y] * k, n_train, n_val,
                                                            q(tx)[sl], y[sl]))
            line = {"size": size, "input_noise": noise, **{k: ce.gmean(v) for k, v in res.items()}}
            rows.append(line)
            print("ESN %3d nodes, input noise %.0e: " % (size, noise)
                  + "   ".join(f"{k} {line[k]:.2e}" for k in res), flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved", args.out + ".csv")


def quad_n(x, size):
    cur = x[:, :size]
    i, j = torch.triu_indices(size, size)
    return torch.cat([x, cur[:, i] * cur[:, j]], 1)


if __name__ == "__main__":
    main()
