"""Screening of readout ideas on cached spectra (no new simulation): multi-ring features and noise-cancelling
features.

Uses the cached central comb lines of the noisy Mackey-Glass comparison (compare_encodings.py,
--noise realistic, group comb_mask by default). Per series, the operating points are ranked by the
validation error of the standard readout; the readout ideas are then scored cross-run (fitted on one run,
tested on the other run, which has independent noise), with the realistic detection noise added:

  standard        log line powers, 31 lines x 8 taps (the paper's readout)
  ratio           log of each line's share of the total detected power (cancels common-mode intensity
                  and detuning fluctuations to first order), plus the log total as one extra feature
  rings K         standard features of the K best operating points concatenated: K rings with different
                  masks, fed from one input comb (31 K lines x 8 taps)

The numbers are optimistic for every variant alike (points ranked on these runs); they serve to compare
variants, not as final results.

    python experiments/readout_variants.py
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import compare_encodings as ce  # noqa: E402
from combrc import make_task  # noqa: E402
from combrc.evaluate import WARMUP, held_out  # noqa: E402
from combrc.readout import add_taps, detector_floor, measure, select_lines, to_features  # noqa: E402


def detected(lines, cfg, seed):
    detector, shot_time = ce.detection(cfg, "realistic")
    gen = torch.Generator().manual_seed(seed)
    return measure(lines, detector, gen, shot_time=shot_time), detector_floor(detector)


def features(lines, cfg, kind, seed, n_lines=31, taps=8):
    p, floor = detected(select_lines(lines, n_lines), cfg, seed)
    if kind == "standard":
        x = to_features(p, floor=floor)
    elif kind == "ratio":
        total = p.sum(-1, keepdim=True)
        x = torch.cat([torch.log10((p + floor) / (total + floor)), torch.log10(total + floor)], -1)
    return add_taps(x, taps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="mackey_glass_h10")
    ap.add_argument("--group", default="comb_mask")
    ap.add_argument("--trials-csv", default="results/robust_noisy_mg_trials.csv")
    ap.add_argument("--series-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--cache", default="results/encodings_spectra")
    args = ap.parse_args()
    import csv
    rows = [r for r in csv.DictReader(open(args.trials_csv)) if r["group"] == args.group and r["task"] == args.task]
    out = {}
    for seed in args.series_seeds:
        y = torch.tensor(make_task(args.task, WARMUP + ce.N_TOTAL, seed=seed)[1], dtype=torch.float64)[WARMUP:]
        mine = sorted([r for r in rows if int(r["series_seed"]) == seed], key=lambda r: float(r["val_31x8"]))
        cfgs = [ce.sample(args.group, int(r["trial"])) for r in mine[: args.top]]
        lines = [ce.spectra_of(args.task, c, 512, 20, args.cache, seed, "realistic")[0] for c in cfgs]

        def score(x):
            return held_out(x[0], y, x_test=x[1])[1]
        # single rings: best by validation of the standard readout, for both readouts
        for kind in ("standard", "ratio"):
            vals = []
            for c, ln in zip(cfgs, lines):
                x = features(ln, c, kind, seed=11)
                v, t, _ = held_out(x[0], y, x_test=x[1])
                vals.append((v, t))
            out.setdefault(f"1 ring, {kind}", []).append(min(vals)[1])
        for k in (2, 4, 8):
            x = torch.cat([features(ln, c, "standard", seed=11 + i) for i, (c, ln) in enumerate(zip(cfgs[:k], lines[:k]))],
                          -1)
            out.setdefault(f"{k} rings, standard", []).append(score(x))
            x = torch.cat([features(ln, c, "ratio", seed=11 + i) for i, (c, ln) in enumerate(zip(cfgs[:k], lines[:k]))],
                          -1)
            out.setdefault(f"{k} rings, ratio", []).append(score(x))
        print("series", seed, {k: f"{v[-1]:.2e}" for k, v in out.items()}, flush=True)
    print(f"\n{args.task}, {args.group}, realistic noise, cross-run test NMSE (geometric mean over series)")
    for k, v in out.items():
        print(f"  {k:22s} {ce.gmean(v):.2e}")


if __name__ == "__main__":
    main()
