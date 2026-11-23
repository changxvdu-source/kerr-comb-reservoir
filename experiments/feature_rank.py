"""Why the random spectral phase helps: how diverse are the 31 line features?

For the mask ablation groups (uniform comb, random spectrum with one common modulation, random amplitudes in
phase, random phases at equal amplitude, full mask), simulate the ring without noise on Mackey-Glass and measure
the diversity of the 31 drop-port line features (log10 powers, standardised):
  participation ratio  (sum of eigenvalues)^2 / sum of squared eigenvalues of their correlation matrix
                       (1: all features move together; 31: all independent)
  mean |correlation|   average absolute off-diagonal correlation.
At the point chosen for each group in the realistic-noise ablation (results/robust_ablation_mg_summary.csv,
``robust_trial``), and over the first ``--draws`` random operating points of each group (the same draws for every
run of the script).

    python experiments/feature_rank.py
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import compare_encodings as ce  # noqa: E402
from combrc import make_task, select_lines, to_features  # noqa: E402
from combrc.evaluate import WARMUP, simulate  # noqa: E402

GROUPS = ["comb_uniform", "comb_static", "comb_amplitude", "comb_phase", "comb_mask"]


def diversity(x):
    """(participation ratio, mean |off-diagonal correlation|) of the columns of x [T, F]; constant columns dropped."""
    x = x[:, x.std(0) > 1e-9]
    c = np.corrcoef(x.T)
    lam = np.clip(np.linalg.eigvalsh(c), 0, None)
    off = np.abs(c[~np.eye(len(c), dtype=bool)])
    return float(lam.sum() ** 2 / (lam ** 2).sum()), float(off.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=16)
    ap.add_argument("--n-modes", type=int, default=256)
    ap.add_argument("--symbols", type=int, default=1000)
    args = ap.parse_args()
    torch.set_num_threads(4)
    chosen = {r["group"]: int(r["robust_trial"]) for r in
              csv.DictReader(open(os.path.join("results", "robust_ablation_mg_summary.csv")))}
    u = torch.tensor(make_task("mackey_glass_h10", WARMUP + args.symbols, seed=0)[0], dtype=torch.float64)
    rows = []
    for g in GROUPS:
        trials = [("chosen", chosen[g])] + [("draw", t) for t in range(args.draws)]
        res = {"chosen": [], "draw": []}
        for kind, t in trials:
            ring = ce.build(ce.sample(g, t), args.n_modes, 20)
            spectra = simulate(ring, u, warmup=WARMUP, n=args.symbols, seeds=(0,))
            x = to_features(select_lines(spectra, 31)[0], floor=1e-9).numpy()
            res[kind].append(diversity(x))
        pr_c, corr_c = res["chosen"][0]
        prs = np.array([p for p, _ in res["draw"]] or [np.nan])
        corrs = np.array([c for _, c in res["draw"]] or [np.nan])
        rows.append({"group": g, "chosen_trial": chosen[g], "chosen_pr": pr_c, "chosen_mean_abs_corr": corr_c,
                     "draws": args.draws, "median_pr": float(np.median(prs)),
                     "median_mean_abs_corr": float(np.median(corrs))})
        print(f"{g:15s} chosen PR {pr_c:5.2f} |r| {corr_c:.2f}   draws: median PR {np.median(prs):5.2f} "
              f"|r| {np.median(corrs):.2f}", flush=True)
    with open(os.path.join("results", "feature_rank.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
