"""How much do the noisy Mackey-Glass results depend on the assumed detuning jitter?

compare_encodings.py --noise realistic assumes a white (per-symbol) detuning jitter of 1 % of the cavity
half-linewidth, and the noise breakdown (floor_breakdown.py) found it to set about half of the error floor.
Here the jitter is swept while everything else stays realistic (intensity noise, quantum field noise,
shot noise, 0.3 uW detectors). For each level, group and series, the operating point is re-chosen the
way the paper's protocol chooses it: among the best ``--select-top`` candidates of the 1 % run
(``--trials-csv``) by the mean validation error on new runs at this level, then reported on further
new runs. The ESN does not see detuning jitter, so its number is the same at every level.

    python experiments/detuning_sweep.py --device cuda
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
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.001, 0.003, 0.01, 0.03],
                    help="rms detuning jitter in units of the half-linewidth")
    ap.add_argument("--groups", nargs="+", default=["comb_mask", "mask_drop", "mask_slow2"])
    ap.add_argument("--task", default="mackey_glass_h10")
    ap.add_argument("--trials-csv", default="results/floor_mg_pin_trials.csv")
    ap.add_argument("--series-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--select-top", type=int, default=16)
    ap.add_argument("--select-pairs", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--variant", default="31x8")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/detuning_sweep")
    args = ap.parse_args()
    variant = tuple(int(x) for x in args.variant.split("x"))
    key = f"val_{args.variant}"
    with open(args.trials_csv) as f:
        rows = [r for r in csv.DictReader(f) if r["task"] == args.task]

    def candidates(group, seed):
        mine = [r for r in rows if r["group"] == group and int(r["series_seed"]) == seed]
        mine.sort(key=lambda r: float(r[key]))
        return [ce.sample(group, int(r["trial"])) for r in mine[: args.select_top]]

    esn = []
    for seed in args.series_seeds:
        u, y = (torch.tensor(a, dtype=torch.float64) for a in make_task(args.task, WARMUP + ce.N_TOTAL, seed=seed))
        esn.append(ce.esn_fresh(u, y, variant[0], variant[1], ce.REALISTIC["baseline_input_noise"], args.pairs,
                                args.select_pairs)[0])
    esn_test = ce.gmean(esn)
    print(f"ESN (no detuning jitter to see): {esn_test:.2e}", flush=True)

    out = []
    for level in args.levels:
        ce.REALISTIC["detuning"] = level
        for group in args.groups:
            tests, picks = [], []
            for seed in args.series_seeds:
                y = torch.tensor(make_task(args.task, WARMUP + ce.N_TOTAL, seed=seed)[1], dtype=torch.float64)[WARMUP:]
                chosen = ce.select_robust(args.task, {group: candidates(group, seed)}, 512, 20, seed, "realistic",
                                          args.select_pairs, args.device, y, variant)
                cfg = chosen[group][0]
                lines = ce.fresh_lines(args.task, [cfg], 512, 20, seed, "realistic", args.pairs, args.device)[0]
                tests.append(np.mean([t for t, _ in ce.fresh_tests(lines, y, variant, "realistic", cfg=cfg)]))
                picks.append(cfg["trial"])
            entry = {"detuning_jitter": level, "group": group, "test": ce.gmean(tests), "esn": esn_test,
                     "trials": " ".join(map(str, picks))}
            out.append(entry)
            print(f"jitter {level:.3%}  {group:11s} {entry['test']:.2e}   (ESN {esn_test:.2e}; points {picks})",
                  flush=True)

    with open(args.out + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(f"\n{args.task}, cross-run test NMSE ({args.variant}) on new runs, geometric mean over "
          f"{len(args.series_seeds)} series; ESN {esn_test:.2e}")
    print(f"{'jitter':>8s}" + "".join(f"{g:>13s}" for g in args.groups))
    for level in args.levels:
        print(f"{level:8.1%}" + "".join(f"{e['test']:13.2e}" for e in out if e["detuning_jitter"] == level))
    print("saved", args.out + ".csv")


if __name__ == "__main__":
    main()
