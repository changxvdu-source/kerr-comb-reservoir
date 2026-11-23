"""Re-choose the operating point for a given pump laser: does a lower Q pay off with a noisy laser?

With a laser of white-noise linewidth dnu the detuning jitter in half-linewidths grows with Q
(eps = 2 Q / nu * sqrt(dnu / (2 pi T))), so the best operating point under the paper's 1 % assumption
(trial 211, Q 1.9e6) need not be the best under a real laser. This repeats the paper's choice of operating
point under the laser's noise (experiments/laser_noise.py), for one group:

  candidates   the 256 random points of ``--group`` (results/floor_mg_pin_trials.csv) in Q bins, the
               ``--per-bin`` best of each bin by their 1 % validation error
  screen       one pair of new runs each, ``--screen-modes`` modes; the ``--top`` best by validation error
  choose       ``--select-pairs`` pairs of new runs each, the lowest mean validation error
  report       ``--pairs`` further pairs of the chosen point: test NMSE
All points in run r see the same laser (the same frequency record), each scaled by its own half-linewidth.
Readout 31 lines x 8 taps, fitted on one run of a pair and scored on the other.

    python experiments/laser_reselect.py --linewidth 1e5 --series-seeds 0 --device cuda
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import compare_encodings as ce  # noqa: E402
import reference_ring as rr  # noqa: E402
from laser_noise import NU, laser_frequency  # noqa: E402
from combrc import make_task, select_lines  # noqa: E402
from combrc.evaluate import WARMUP, held_out  # noqa: E402
from combrc.lle import stack_rings  # noqa: E402

Q_BINS = (0, 7e5, 1e6, 1.5e6, 2e6, 3e6, 5e6, 1e9)


@torch.no_grad()
def lines_of(cfgs, u, nu, n_modes, steps, device, seed, block=200, max_rows=240):
    """Central lines [len(cfgs)][runs, symbols, 31] of every point; run r of every point sees laser record nu[r]."""
    runs = len(nu)
    out, families = [None] * len(cfgs), {}
    for i, cfg in enumerate(cfgs):
        families.setdefault(ce.family(cfg), []).append(i)
    per_batch = max(1, max_rows // runs)
    for members in families.values():
        for k in range(0, len(members), per_batch):
            part = members[k: k + per_batch]
            rings = [ce.build(cfgs[i], n_modes, steps) for i in part]
            hwhm = torch.tensor([NU / (2 * float(r.q_total)) for r in rings], dtype=torch.float64)
            eps = (torch.tensor(nu)[None] / hwhm[:, None, None]).reshape(len(part) * runs, -1)   # ring-major rows
            stack = ce.with_noise(stack_rings(rings, repeats=runs).to(device), "realistic")
            state = torch.cat([r.init_state(1, seed=100 + s) for r in rings for s in range(runs)]).to(device)
            gen = torch.Generator(device=device).manual_seed(5_000_011 + 7919 * seed + 104_729 * k)
            kept = []
            for t0 in range(0, len(u), block):
                spectra, state = stack(u[t0: t0 + block].to(device).expand(len(part) * runs, -1), state=state,
                                       generator=gen, detuning=eps[:, t0: t0 + block].to(device))
                if t0 + block > WARMUP:
                    kept.append(select_lines(spectra[:, max(WARMUP - t0, 0):], rr.LINES).cpu())
            lines = torch.cat(kept, 1)
            for j, i in enumerate(part):
                out[i] = lines[j * runs: (j + 1) * runs].clone()
            del stack, state, lines, kept
    return out


def scores(lines, y, cfg, seed):
    """(validation, test) NMSE of each pair of runs."""
    if not torch.isfinite(lines).all():
        return [(float("inf"), float("inf"))] * (len(lines) // 2)
    out = []
    for k in range(len(lines) // 2):
        gen = torch.Generator().manual_seed(seed + k)
        xa = rr.features(lines[2 * k], None, cfg, gen, "lines")
        xb = rr.features(lines[2 * k + 1], None, cfg, gen, "lines")
        try:
            val, test, _ = held_out(xa, y, x_test=xb, n_test=ce.N_TEST)
        except (RuntimeError, FloatingPointError):
            val, test = float("inf"), float("inf")
        out.append((val, test))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--linewidth", type=float, required=True)
    ap.add_argument("--group", default="mask_drop")
    ap.add_argument("--series-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--per-bin", type=int, default=8)
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--select-pairs", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--screen-modes", type=int, default=256)
    ap.add_argument("--n-modes", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--trials-csv", default="results/floor_mg_pin_trials.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(4)
    ce.REALISTIC["detuning"] = 1.0                       # the supplied sequences are the jitter itself
    with open(args.trials_csv) as f:
        trials = [r for r in csv.DictReader(f) if r["task"] == "mackey_glass_h10" and r["group"] == args.group]
    n = WARMUP + ce.N_TOTAL
    rows = []
    for seed in args.series_seeds:
        mine = [r for r in trials if int(r["series_seed"]) == seed]
        cands = []
        for lo, hi in zip(Q_BINS, Q_BINS[1:]):
            inbin = sorted((r for r in mine if lo <= float(r["q_total"]) < hi), key=lambda r: float(r["val_31x8"]))
            cands += [int(r["trial"]) for r in inbin[: args.per_bin]]
        cfgs = [ce.sample(args.group, t) for t in cands]
        u_np, y_np = make_task("mackey_glass_h10", n, seed=seed)
        u = torch.tensor(u_np, dtype=torch.float64)[:n]
        y = torch.tensor(y_np, dtype=torch.float64)[WARMUP: n]

        def laser(runs, salt):
            return laser_frequency(runs, n, 1e-9, args.linewidth, 0.0, 0.0, 2 ** 15, seed=10_007 * seed + salt)

        # screen
        nu = laser(2, 1)
        lines = lines_of(cfgs, u, nu, args.screen_modes, args.steps, args.device, seed)
        screen = [scores(l, y, c, 2000)[0][0] for l, c in zip(lines, cfgs)]
        order = np.argsort(screen)[: args.top]
        top = [cfgs[i] for i in order]
        # choose
        nu = laser(2 * args.select_pairs, 2)
        lines = lines_of(top, u, nu, args.n_modes, args.steps, args.device, seed)
        means = [float(np.mean([v for v, _ in scores(l, y, c, 5000)])) for l, c in zip(lines, top)]
        best = top[int(np.argmin(means))]
        # report
        nu = laser(2 * args.pairs, 3)
        lines = lines_of([best], u, nu, args.n_modes, args.steps, args.device, seed)[0]
        test = float(np.mean([t for _, t in scores(lines, y, best, 1000)]))
        q = float(ce.build(best, args.n_modes, args.steps).q_total)
        rows.append({"linewidth_hz": args.linewidth, "series": seed, "trial": best["trial"], "q_total": q,
                     "p_total_mw": best["p_total"] * 1e3,
                     "jitter_rms": math.sqrt(args.linewidth / (2 * math.pi * 1e-9)) / (NU / (2 * q)),
                     "test_nmse": test, "candidates": len(cfgs),
                     "screen_q_of_top": " ".join(f"{float(ce.build(c, 64, 4).q_total):.1e}" for c in top[:5])})
        print(f"linewidth {args.linewidth:g} Hz, series {seed}: chose trial {best['trial']} (Q {q:.2e}, "
              f"{best['p_total'] * 1e3:.0f} mW, jitter {rows[-1]['jitter_rms']:.2%}), test NMSE {test:.2e}", flush=True)
    out = args.out or f"results/laser_reselect_lw{args.linewidth:g}"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
