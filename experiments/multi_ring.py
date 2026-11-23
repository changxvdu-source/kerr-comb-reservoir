"""Several rings with different masks fed from one input comb, readouts concatenated, under the paper's
fresh-run protocol.

Per group and series, the operating points are ranked by the validation error of the standard readout on
the cached runs of the comparison (compare_encodings.py; ``--trials-csv``). The K best are then
re-simulated on ``--pairs`` pairs of new runs with independent noise, and their detected features
(31 lines x 8 taps each, realistic detection noise) are concatenated: the readout is fitted on one run of
a pair and tested on the other. The choice of the points never sees these runs.

    python experiments/multi_ring.py --device cuda
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
from combrc.evaluate import WARMUP, held_out  # noqa: E402
from combrc.readout import add_taps, detector_floor, measure, select_lines, to_features  # noqa: E402
from combrc.tasks import is_symbol_task, symbol_error_rate  # noqa: E402


def features(lines, cfg, seed, n_lines=31, taps=8):
    detector, shot_time = ce.detection(cfg, "realistic")
    gen = torch.Generator().manual_seed(seed)
    p = measure(select_lines(lines, n_lines), detector, gen, shot_time=shot_time)
    return add_taps(to_features(p, floor=detector_floor(detector)), taps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="mackey_glass_h10")
    ap.add_argument("--groups", nargs="+", default=["comb_mask", "mask_drop", "mask_slow2"])
    ap.add_argument("--trials-csv", nargs="+", default=["results/robust_noisy_mg_trials.csv",
                                                        "results/floor_mg_pin_trials.csv"])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--series-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--rank-task", default=None,
                    help="rank operating points by this task's trials (default: --task), e.g. the 28 dB channel for 40 dB")
    ap.add_argument("--taps", type=int, default=8)
    ap.add_argument("--rank-variant", default="31x8", help="readout variant whose validation error ranks the points")
    ap.add_argument("--test-symbols", type=int, default=0, help="test part length (default: last 20 %%)")
    ap.add_argument("--fit-symbols", type=int, default=0,
                    help="training + validation length (default 1360; needs --test-symbols)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ridge-device", default="cpu", help="where the readouts are fitted ('cuda' for many features)")
    ap.add_argument("--shared-rin", action="store_true",
                    help="the rings share the source intensity noise (one laser feeds them all)")
    ap.add_argument("--out", default="results/multi_ring")
    args = ap.parse_args()
    ce.set_length(args.test_symbols, None, args.fit_symbols or None)
    error_rate = symbol_error_rate if is_symbol_task(args.task) else None
    rows = []
    for name in args.trials_csv:
        rows += [r for r in csv.DictReader(open(name)) if r["task"] == (args.rank_task or args.task)]
    out = []
    for group in args.groups:
        per = {k: [] for k in args.ks}
        per_ser = {k: [] for k in args.ks}
        for seed in args.series_seeds:
            y = torch.tensor(make_task(args.task, WARMUP + ce.N_TOTAL, seed=seed)[1], dtype=torch.float64)[WARMUP:]
            mine = {int(r["trial"]): r for r in rows if r["group"] == group and int(r["series_seed"]) == seed}
            ranked = sorted(mine.values(), key=lambda r: float(r[f"val_{args.rank_variant}"]))[: max(args.ks)]
            cfgs = [ce.sample(group, int(r["trial"])) for r in ranked]
            lines = ce.fresh_lines(args.task, cfgs, 512, 20, seed, "realistic", args.pairs, args.device, salt=3,
                                   shared_rin=args.shared_rin)
            for k in args.ks:
                tests, sers = [], []
                for p in range(args.pairs):
                    x = torch.cat([features(lines[i][2 * p: 2 * p + 2].double(), cfgs[i], seed=1000 + 31 * p + i,
                                            taps=args.taps) for i in range(k)], -1)
                    x, yd = x.to(args.ridge_device), y.to(args.ridge_device)
                    _, test, pred = held_out(x[0], yd, x_test=x[1], n_test=ce.N_TEST)
                    tests.append(test)
                    if error_rate:
                        sers.append(error_rate(y[len(y) - len(pred):].numpy(), pred.cpu().numpy()))
                per[k].append(float(np.mean(tests)))
                per_ser[k].append(float(np.mean(sers)) if sers else float("nan"))
            print(group, "series", seed, {k: f"{v[-1]:.2e}" for k, v in per.items()}, flush=True)
        for k in args.ks:
            out.append({"task": args.task, "group": group, "rings": k, "lines": 31 * k,
                        "fit_symbols": ce.N_TOTAL - (ce.N_TEST or 0) if ce.N_TEST else int(0.8 * ce.N_TOTAL),
                        "test_nmse": ce.gmean(per[k]), "ser": float(np.mean(per_ser[k])),
                        "per_series": " ".join(f"{v:.3e}" for v in per[k])})
            print(f"  {group:11s} {k} rings ({31 * k} lines): {out[-1]['test_nmse']:.2e}"
                  + (f"  SER {out[-1]['ser']:.2e}" if error_rate else ""), flush=True)
    with open(args.out + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print("saved", args.out + ".csv")


if __name__ == "__main__":
    main()
