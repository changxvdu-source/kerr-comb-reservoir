"""Optimise the input mask at a fixed operating point: random search, then a simple evolution strategy.

The mask (complex bias and signal weight of each of the 16 input lines) is set by a programmable spectral shaper,
so it can be chosen like a trained parameter. Physical parameters stay at a given point (``--trial`` of mask_drop,
optionally with a new detuning ``--zeta``). Scheme as in counterprop_comb.py (bi-diff, uni62, ...).

  generation 0   ``--population`` random masks, scored on ``--pairs`` pairs of runs of series 0 (validation NMSE)
  generations    ``--elite`` best masks are kept; each child = an elite mask + sigma * complex normal noise
                 (relative to the rms entry), sigma shrinking by ``--decay`` per generation; children re-scored
                 on new runs of series 0 (the elite is re-scored with them, so a lucky score does not stick)
  choice         the ``--final`` best masks scored on 3 fresh pairs on each of series 0-2 (validation); best kept
  report         that mask on ``--report-pairs`` fresh pairs on held-out series 3-5 (never used above), with and
                 without the detuning jitter; the original random mask of the point is reported the same way

Every score is cross-run (fitted on one run, tested on another with independent noise), realistic noise.

    python experiments/mask_optimize.py --device cuda --scheme bi-diff --trial 149 --zeta 1.73
"""
import argparse
import csv
import json
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

K = len(ce.input_lines("mask_drop"))


def cnormal(shape, gen):
    return torch.complex(*torch.randn(2, *shape, generator=gen, dtype=torch.float64))


def original_mask(trial):
    gen = torch.Generator().manual_seed(trial)
    return tuple(cnormal((K,), gen) for _ in range(2))


def score(base, masks, scheme, args, seeds, pairs, salt, jitter=True):
    """Mean validation and test NMSE per mask (geometric mean over ``seeds``), cross-run."""
    R = 2 * pairs
    U = torch.cat([cc.task(s)[0].unsqueeze(0).expand(R, -1) for s in seeds])
    jf = torch.full((len(U),), 1.0 if jitter else 0.0)
    rows = len(U)
    per = max(1, args.max_rows // rows)
    vals, tests = [], []
    for i in range(0, len(masks), per):
        cfgs = [{**base, "mask_raw": m} for m in masks[i: i + per]]
        L = cc.simulate(cfgs, cc.SIM[scheme], U, rows, args.device, salt, args.n_modes, args.steps, jitter=jf)
        for j, c in enumerate(cfgs):
            mine = L[j * rows: (j + 1) * rows]
            v, t = [], []
            for k, s in enumerate(seeds):
                sc = cc.pair_scores(mine[k * R: (k + 1) * R], scheme, c, cc.task(s)[1], k0=salt)
                v.append(float(np.mean([a for a, _ in sc])))
                t.append(float(np.mean([b for _, b in sc])))
            vals.append(ce.gmean(v))
            tests.append(ce.gmean(t))
    return vals, tests


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scheme", default="bi-diff", choices=cc.SCHEMES)
    ap.add_argument("--trial", type=int, default=149)
    ap.add_argument("--zeta", type=float, default=None)
    ap.add_argument("--population", type=int, default=512)
    ap.add_argument("--children", type=int, default=240)
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--elite", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--decay", type=float, default=0.75)
    ap.add_argument("--pairs", type=int, default=1)
    ap.add_argument("--final", type=int, default=8)
    ap.add_argument("--report-pairs", type=int, default=3)
    ap.add_argument("--n-modes", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--max-rows", type=int, default=960)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/mask_optimize")
    args = ap.parse_args()
    t0 = time.time()
    base = {k: v for k, v in ce.sample("mask_drop", args.trial).items() if k in ce.CFG_KEYS}
    if args.zeta is not None:
        base = {**base, "zeta": args.zeta, "delta0": args.zeta * ce.cavity(base["log_q_ext1"], base["log_q_ext2"])[0]}
    gen = torch.Generator().manual_seed(12345 + args.seed)
    log = []

    def note(msg):
        print(f"[{time.time() - t0:6.0f} s] {msg}", flush=True)

    pop = [tuple(cnormal((K,), gen) for _ in range(2)) for _ in range(args.population)]
    vals, _ = score(base, pop, args.scheme, args, [0], args.pairs, salt=400)
    order = np.argsort(vals)
    elite = [pop[i] for i in order[: args.elite]]
    note(f"generation 0: best {vals[order[0]]:.2e}, elite worst {vals[order[args.elite - 1]]:.2e}")
    log.append({"generation": 0, "best_val": float(vals[order[0]]), "median_val": float(np.median(vals))})
    sigma = args.sigma
    for g in range(1, args.generations + 1):
        children = []
        for i in range(args.children):
            b, w = elite[i % len(elite)]
            rb, rw = b.abs().pow(2).mean().sqrt(), w.abs().pow(2).mean().sqrt()
            children.append((b + sigma * rb * cnormal((K,), gen), w + sigma * rw * cnormal((K,), gen)))
        cand = elite + children
        vals, _ = score(base, cand, args.scheme, args, [0], args.pairs, salt=400 + 10 * g)
        order = np.argsort(vals)
        elite = [cand[i] for i in order[: args.elite]]
        note(f"generation {g} (sigma {sigma:.3f}): best {vals[order[0]]:.2e}, elite worst {vals[order[args.elite - 1]]:.2e}")
        log.append({"generation": g, "best_val": float(vals[order[0]]), "median_val": float(np.median(vals))})
        sigma *= args.decay
    finals = elite[: args.final]
    vals, _ = score(base, finals, args.scheme, args, [0, 1, 2], 3, salt=700)
    best = finals[int(np.argmin(vals))]
    note(f"choice on series 0-2: {min(vals):.2e} (validation)")
    report = {}
    for label, mask in (("optimised", best), ("original", original_mask(args.trial))):
        for jl, jit in (("jitter", True), ("no jitter", False)):
            _, t = score(base, [mask], args.scheme, args, [3, 4, 5], args.report_pairs, salt=900, jitter=jit)
            report[(label, jl)] = t[0]
        note(f"{label:9s} mask on held-out series 3-5: {report[(label, 'jitter')]:.2e} with jitter, "
             f"{report[(label, 'no jitter')]:.2e} without")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mask", "jitter", "test_nmse_heldout_series_3_5"])
        for (label, jl), v in report.items():
            w.writerow([label, jl, v])
    with open(args.out + "_log.json", "w") as f:
        json.dump({"args": vars(args), "generations": log,
                   "best_mask": {"bias": [[z.real, z.imag] for z in best[0].tolist()],
                                 "weight": [[z.real, z.imag] for z in best[1].tolist()]}}, f, indent=1)
    note("saved " + args.out + ".csv")


if __name__ == "__main__":
    main()
