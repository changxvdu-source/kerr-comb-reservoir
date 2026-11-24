"""Idea A: is there a detuning "sweet spot" where the ring is insensitive to laser detuning jitter?

At the top of a resonance the transmitted power is stationary in the detuning, so detuning jitter enters only in
second order. With the Kerr shift and dispersion this cannot hold for every input and every line, but a region of
reduced jitter sensitivity might exist. This maps, around the best single ring on noisy Mackey-Glass (mask_drop
trial 211: same mask, couplings and depth), the cross-run test error over a grid of detuning zeta and total power,
with realistic noise once with and once without the detuning jitter (all other noise on). Readout: the paper's
(31 lines x 8 taps, linear ridge, fitted on one run and tested on the other of a pair).

Output per grid point: error with jitter, error without, and their ratio (the jitter penalty). A sweet spot is a
region where the penalty is near 1 while the error without jitter stays low.

    python experiments/jitter_sweet_spot.py --device cuda --series-seeds 0
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
from combrc import make_task  # noqa: E402
from combrc.evaluate import WARMUP  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="mask_drop")
    ap.add_argument("--trial", type=int, default=211)
    ap.add_argument("--zetas", type=float, nargs=3, default=[-1.0, 4.0, 26], metavar=("FROM", "TO", "N"))
    ap.add_argument("--powers", type=float, nargs="+", default=[0.5, 1.0, 1.5], help="factors on the point's power")
    ap.add_argument("--series-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--n-modes", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/jitter_sweet_spot")
    args = ap.parse_args()
    base = ce.sample(args.group, args.trial)
    alpha = base["delta0"] / base["zeta"]
    zetas = np.linspace(args.zetas[0], args.zetas[1], int(args.zetas[2]))
    cfgs = []
    for f in args.powers:
        for z in zetas:
            c = {k: base[k] for k in ce.CFG_KEYS if k in base}
            c.update(zeta=float(z), delta0=float(z) * alpha, p_total=base["p_total"] * f)
            cfgs.append(c)
    print(f"{len(cfgs)} grid points around {args.group} trial {args.trial} "
          f"(zeta {base['zeta']:.2f}, {base['p_total'] * 1e3:.0f} mW)", flush=True)
    jit = ce.REALISTIC["detuning"]
    res = {}
    for seed in args.series_seeds:
        _, y = make_task("mackey_glass_h10", WARMUP + ce.N_TOTAL, seed=seed)
        y = torch.tensor(y, dtype=torch.float64)[WARMUP:]
        for label, level in (("jitter", jit), ("no jitter", 0.0)):
            ce.REALISTIC["detuning"] = level
            lines = ce.fresh_lines("mackey_glass_h10", cfgs, args.n_modes, args.steps, seed, "realistic",
                                   args.pairs, args.device, salt=7)
            for i, c in enumerate(cfgs):
                t = [x for x, _ in ce.fresh_tests(lines[i], y, (31, 8), "realistic", cfg=c)]
                res.setdefault((i, label), []).append(float(np.mean(t)))
            print(f"series {seed} {label} done", flush=True)
        ce.REALISTIC["detuning"] = jit
    rows = []
    for i, c in enumerate(cfgs):
        a, b = ce.gmean(res[(i, "jitter")]), ce.gmean(res[(i, "no jitter")])
        rows.append({"zeta": c["zeta"], "p_mw": c["p_total"] * 1e3, "nmse_jitter": a, "nmse_no_jitter": b,
                     "jitter_penalty": a / b if b > 0 else float("nan")})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    best = sorted(rows, key=lambda r: r["nmse_jitter"])[:8]
    print("best points with jitter:")
    for r in best:
        print(f"  zeta {r['zeta']:5.2f}  {r['p_mw']:5.0f} mW  jitter {r['nmse_jitter']:.2e}  "
              f"no jitter {r['nmse_no_jitter']:.2e}  penalty {r['jitter_penalty']:.2f}")
    print("saved", args.out + ".csv")


if __name__ == "__main__":
    main()
