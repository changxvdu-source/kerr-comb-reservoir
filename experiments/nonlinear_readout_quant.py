"""Does the quadratic readout's gain survive digitisation of the ring's 31 readings?

Reuses the line powers cached by readout_quantization.py (best single ring on noisy Mackey-Glass, mask_drop
trial 211, realistic noise, 12 runs with independent noise per series; ``results/readout_quantization_lines_s*.npy``)
and its ADC models (per line, full scale calibrated once on the first fitting run; ``linear`` digitises the
photocurrent, ``log`` has a logarithmic amplifier before the ADC). Readouts as in nonlinear_readout.py: ridge on
31 lines x 8 taps (linear) or with all products of the current 31 features added (quadratic), fitted on the first
K runs and tested on each of the last ``--test-runs`` runs (mean over them).

Software side, as in the paper's Table 3: the 31-node ESN with its input quantized to N bits and no other noise,
with the same two readouts.

    python experiments/nonlinear_readout_quant.py
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
import nonlinear_readout as nr  # noqa: E402
import readout_quantization as rq  # noqa: E402
from combrc import make_task  # noqa: E402
from combrc.evaluate import WARMUP  # noqa: E402
from combrc.readout import detector_floor  # noqa: E402

BITS = [None, 6, 7, 8, 9, 10, 12]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="mask_drop")
    ap.add_argument("--trial", type=int, default=211)
    ap.add_argument("--series-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--fit-runs", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--test-runs", type=int, default=4)
    ap.add_argument("--modes", nargs="+", default=["log", "linear"])
    ap.add_argument("--cache", default="results/readout_quantization")
    ap.add_argument("--out", default="results/nonlinear_readout_quant")
    args = ap.parse_args()
    torch.set_num_threads(4)
    cfg = ce.sample(args.group, args.trial)
    floor = detector_floor(ce.detection(cfg, "realistic")[0])
    n = ce.N_TOTAL
    n_train, n_val = int(0.6 * n), int(0.2 * n)
    sl = slice(n_train + n_val, n)
    keys = [(sys_, mode, b, r, k) for sys_, modes in (("ring", args.modes), ("ESN 31", ["input"]))
            for mode in modes for b in BITS for r in ("linear", "quadratic") for k in args.fit_runs]
    keys = [kk for kk in keys if not (kk[0] == "ring" and kk[1] != args.modes[0] and kk[2] is None)]
    res = {kk: [] for kk in keys}
    for seed in args.series_seeds:
        u_np, y_np = make_task("mackey_glass_h10", WARMUP + n, seed=seed)
        u = torch.tensor(u_np, dtype=torch.float64)[: WARMUP + n]
        y = torch.tensor(y_np, dtype=torch.float64)[WARMUP: WARMUP + n]
        lines = torch.tensor(np.load(f"{args.cache}_lines_s{seed}.npy"))
        runs = lines.shape[0]
        p = [rq.readings(lines[r], cfg, torch.Generator().manual_seed(1000 + 97 * seed + r)) for r in range(runs)]
        tests = list(range(runs - args.test_runs, runs))
        for (sys_, mode, b, r, k) in keys:
            if sys_ == "ring":
                feats = {i: rq.features(p[i], p[0], b, mode, floor) for i in list(range(max(args.fit_runs))) + tests}
            else:
                uq = u if b is None else rq.quantize(u, 0.0, 1.0, b)
                e = nr.esn_runs(uq, y, 1, 0.0, n_train, n_val)[0]
                feats = {i: e for i in list(range(max(args.fit_runs))) + tests}
            tf = nr.quad if r == "quadratic" else (lambda x: x)
            fit = [tf(feats[i]) for i in range(k)]
            errs = [nr.ridge(fit, [y] * k, n_train, n_val, tf(feats[t])[sl], y[sl]) for t in tests]
            res[(sys_, mode, b, r, k)].append(float(np.mean(errs)))
        print(f"series {seed} done", flush=True)
    rows = []
    for kk in keys:
        sys_, mode, b, r, k = kk
        rows.append({"system": sys_, "adc": mode, "bits": "" if b is None else b, "readout": r, "fit_runs": k,
                     "test_nmse": ce.gmean(res[kk]), "per_series": " ".join(f"{x:.3e}" for x in res[kk])})
    for k in args.fit_runs:
        print(f"\n=== fitted on {k} run(s); rows: bits (none = no ADC)")
        cols = [("ring", m, r) for m in args.modes for r in ("linear", "quadratic")] + \
               [("ESN 31", "input", r) for r in ("linear", "quadratic")]
        print("bits   " + "  ".join(f"{s[:4]} {m[:3]} {r[:4]:>4s}" for s, m, r in cols))
        for b in BITS:
            vals = []
            for s, m, r in cols:
                row = [x for x in rows if x["system"] == s and x["adc"] == (args.modes[0] if (s == "ring" and b is None) else m)
                       and x["bits"] == ("" if b is None else b) and x["readout"] == r and x["fit_runs"] == k]
                vals.append(f"{row[0]['test_nmse']:.2e}" if row else "   -    ")
            print(f"{'none' if b is None else b:>4}   " + "      ".join(vals))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved", args.out + ".csv")


if __name__ == "__main__":
    main()
