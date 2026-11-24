"""Focused search around the best two-direction differential point (counterprop_comb.py, bi-diff, mask_drop trial
149, 2.4e-5 with jitter): is it a robust region or a narrow peak?

One-parameter scans around the point, each other parameter held: detuning zeta (+-0.6), total power (x0.7..1.3),
signal depth (x0.6..1.2, capped below 1), side fraction (+-0.2) and the input coupling log10 Q_ext1 (+-0.2), with the
detuning re-expressed for the new linewidth; plus the same physical parameters with 16 other random masks (does the
point owe its quality to its mask?). Every variant: bi-diff, realistic noise with jitter, ``--pairs`` pairs of new
runs on each of 3 series, cross-run test NMSE of the 62 x 8 readout, geometric mean over the series.

    python experiments/counterprop_focus.py --device cuda
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import compare_encodings as ce  # noqa: E402
import counterprop_comb as cc  # noqa: E402


def variants(base):
    out = [("base", "", 0.0, base)]

    def with_zeta(c, z):
        return {**c, "zeta": z, "delta0": z * ce.cavity(c["log_q_ext1"], c["log_q_ext2"])[0]}

    for dz in np.linspace(-0.6, 0.6, 13):
        if abs(dz) > 1e-9:
            out.append(("zeta", "", float(base["zeta"] + dz), with_zeta(base, base["zeta"] + dz)))
    for f in (0.7, 0.8, 0.9, 0.95, 1.05, 1.1, 1.2, 1.3):
        out.append(("power", "x", f, {**base, "p_total": base["p_total"] * f}))
    for f in (0.6, 0.8, 0.9, 1.1, 1.2):
        out.append(("depth", "x", f, {**base, "depth": min(0.99, base["depth"] * f)}))
    for d in (-0.2, -0.1, -0.05, 0.05, 0.1, 0.2):
        out.append(("side_fraction", "+", d, {**base, "side_fraction": float(np.clip(base["side_fraction"] + d, 0.05, 0.95))}))
    for d in (-0.2, -0.1, -0.05, 0.05, 0.1, 0.2):
        out.append(("log_q_ext1", "+", d, with_zeta({**base, "log_q_ext1": base["log_q_ext1"] + d}, base["zeta"])))
    for m in range(16):
        out.append(("mask", "seed", float(m), {**base, "mask_seed": 500_000 + m}))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trial", type=int, default=149)
    ap.add_argument("--scheme", default="bi-diff")
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--series-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-modes", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--max-rows", type=int, default=480)
    ap.add_argument("--limit", type=int, default=0, help="only the first N variants (smoke test)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/counterprop_focus")
    args = ap.parse_args()
    t0 = time.time()
    base = {k: v for k, v in ce.sample("mask_drop", args.trial).items() if k in ce.CFG_KEYS}
    var = variants(base)
    if args.limit:
        var = var[: args.limit]
    R = 2 * args.pairs
    U = torch.cat([cc.task(s)[0].unsqueeze(0).expand(R, -1) for s in args.series_seeds])
    ys = [cc.task(s)[1] for s in args.series_seeds]
    rows_per = len(U)
    per = max(1, args.max_rows // rows_per)
    sim = cc.SIM[args.scheme]
    rows = []
    for i in range(0, len(var), per):
        chunk = var[i: i + per]
        L = cc.simulate([c for *_, c in chunk], sim, U, rows_per, args.device, 300, args.n_modes, args.steps)
        for j, (param, kind, value, c) in enumerate(chunk):
            mine = L[j * rows_per: (j + 1) * rows_per]
            per_series = []
            for k, y in enumerate(ys):
                sc = cc.pair_scores(mine[k * R: (k + 1) * R], args.scheme, c, y, k0=300)
                per_series.append(float(np.mean([te for _, te in sc])))
            rows.append({"param": param, "kind": kind, "value": value, "zeta": c["zeta"],
                         "p_total_mw": c["p_total"] * 1e3, "depth": c["depth"], "side_fraction": c["side_fraction"],
                         "log_q_ext1": c["log_q_ext1"], "test": ce.gmean(per_series),
                         "per_series": " ".join(f"{x:.3e}" for x in per_series)})
        print(f"[{time.time() - t0:6.0f} s] {min(i + per, len(var))}/{len(var)} variants", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['param']:14s} {r['kind']}{r['value']:<8.3g} test {r['test']:.2e}")
    print("saved", args.out + ".csv")


if __name__ == "__main__":
    main()
