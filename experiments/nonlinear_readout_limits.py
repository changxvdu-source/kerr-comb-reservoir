"""How far can the readout go? Longer memory (more taps), richer products and more training runs, ring vs ESN.

Same cached runs and ADC models as nonlinear_readout_quant.py (best single ring on noisy Mackey-Glass, 12 runs per
series; ring readings through a per-line log-amplifier ADC calibrated on the first fitting run; ESN 31 nodes with
its input quantized to N bits). The ESN gets dithered copies of its quantized input as extra training runs (0.5 LSB
rms), so that both systems see K different noise realisations. Tested on each of the last 4 runs.

Readouts, all ridge (strength on the validation parts of the fitting runs):
  linear      31 features x T taps
  quad        + all products of the current 31 features
  quad-lag    + all products among the current and the previous symbol's 31 features (62 -> 1953 products)

    python experiments/nonlinear_readout_limits.py
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
from combrc import make_task, to_features  # noqa: E402
from combrc.evaluate import WARMUP, esn_states  # noqa: E402
from combrc.readout import add_taps, detector_floor, fit_readout, nmse  # noqa: E402

L = 31


def prods(z):
    i, j = torch.triu_indices(z.shape[1], z.shape[1])
    return z[:, i] * z[:, j]


def expand(base, taps, kind):
    """base: [T, 31] features of one run (no taps)."""
    x = add_taps(base, taps)
    if kind == "linear":
        return x
    if kind == "quad":
        return torch.cat([x, prods(base)], 1)
    lag = torch.cat([base[:1], base[:-1]])
    return torch.cat([x, prods(torch.cat([base, lag], 1))], 1)


def ring_base(p, cal, bits, floor):
    xc, x = to_features(cal, floor=floor), to_features(p, floor=floor)
    if bits is None:
        return x
    lo, hi = xc.min(0).values, xc.max(0).values
    span = (hi - lo).clamp_min(1e-6)
    return rq.quantize(x, lo - 0.05 * span, hi + 0.05 * span, bits)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 10])
    ap.add_argument("--taps", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--kinds", nargs="+", default=["linear", "quad", "quad-lag"])
    ap.add_argument("--fit-runs", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--cache", default="results/readout_quantization")
    ap.add_argument("--out", default="results/nonlinear_readout_limits")
    args = ap.parse_args()
    torch.set_num_threads(4)
    cfg = ce.sample("mask_drop", 211)
    floor = detector_floor(ce.detection(cfg, "realistic")[0])
    n = ce.N_TOTAL
    n_train, n_val = int(0.6 * n), int(0.2 * n)
    sl = slice(n_train + n_val, n)
    res = {}
    for seed in args.series_seeds:
        u_np, y_np = make_task("mackey_glass_h10", WARMUP + n, seed=seed)
        u = torch.tensor(u_np, dtype=torch.float64)[: WARMUP + n]
        y = torch.tensor(y_np, dtype=torch.float64)[WARMUP: WARMUP + n]
        lines = torch.tensor(np.load(f"{args.cache}_lines_s{seed}.npy"))
        runs = lines.shape[0]
        p = [rq.readings(lines[r], cfg, torch.Generator().manual_seed(1000 + 97 * seed + r)) for r in range(runs)]
        tests = list(range(runs - 4, runs))
        fit_idx = list(range(max(args.fit_runs)))
        for b in args.bits:
            ring = {i: ring_base(p[i], p[0], b, floor) for i in fit_idx + tests}
            uq = rq.quantize(u, 0.0, 1.0, b)
            lsb = 1.0 / (2 ** b - 1)

            def st(inp, r, s):
                return esn_states(inp, L, r, s)[WARMUP:]

            def val(r, s):
                x = add_taps(st(uq, r, s), 8)
                m = fit_readout(x, y, n_train, n_val)
                return nmse(y[n_train:n_train + n_val], m.predict(x[n_train:n_train + n_val])).item()

            _, r_, s_ = min((val(r, s), r, s) for r in (0.5, 0.8, 0.95, 1.1) for s in (0.1, 0.3, 1.0))
            esn = {0: st(uq, r_, s_)}
            for i in fit_idx[1:]:
                g = torch.Generator().manual_seed(500 + i)
                esn[i] = st(uq + 0.5 * lsb * torch.randn(u.shape, generator=g, dtype=u.dtype), r_, s_)
            for t in tests:
                esn[t] = esn[0]
            for sysname, feats in (("ring", ring), ("ESN 31", esn)):
                for taps in args.taps:
                    for kind in args.kinds:
                        ex = {i: expand(feats[i], taps, kind) for i in feats}
                        for k in args.fit_runs:
                            fit = [ex[i] for i in range(k)]
                            e = float(np.mean([nr.ridge(fit, [y] * k, n_train, n_val, ex[t][sl], y[sl]) for t in tests]))
                            res.setdefault((sysname, b, taps, kind, k), []).append(e)
            print(f"series {seed}, {b} bits done", flush=True)
    rows = [{"system": s, "bits": b, "taps": t, "readout": kd, "fit_runs": k, "test_nmse": ce.gmean(v),
             "per_series": " ".join(f"{x:.3e}" for x in v)} for (s, b, t, kd, k), v in res.items()]
    for b in args.bits:
        print(f"\n=== {b} bits: test NMSE (ring / ESN / ESN-to-ring ratio)")
        for k in args.fit_runs:
            for t in args.taps:
                cells = []
                for kd in args.kinds:
                    rg, es = ce.gmean(res[("ring", b, t, kd, k)]), ce.gmean(res[("ESN 31", b, t, kd, k)])
                    cells.append(f"{kd:8s} {rg:.2e} / {es:.2e} / {es / rg:4.2f}")
                print(f"K={k} taps={t:2d}:  " + "   ".join(cells))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved", args.out + ".csv")


if __name__ == "__main__":
    main()
