"""What does digitising the ring's 31 readings cost? Readout-side quantization of the best single ring.

The software comparison (paper, section 8) reads the software reservoir's input noise as the quantization noise
of an ADC. The ring's own readings were never quantized there; this closes that asymmetry. Each of the 31
drop-port lines has its own photodiode and its own N-bit ADC.

Operating point: the best single ring on noisy Mackey-Glass (stronger drop coupling, trial 211, 1.55e-4 on new
runs). Realistic noise (intensity noise, 1 % detuning jitter, quantum field noise, shot noise, 0.3 uW
detectors). ``--pairs`` pairs of new runs per series; readout fitted on one run of a pair, tested on the other.

ADC models, per line, full scale calibrated once on the fitting run (an AGC set before use, then fixed):
  linear   the ADC digitises the photocurrent: uniform steps over [0, 1.1 x the largest reading of that line],
           then the logarithm as in the paper's readout. Small lines get few levels.
  log      a logarithmic amplifier before the ADC: uniform steps over the range of the log readings of that line
           (5 % margin on each side), i.e. equal relative steps.
For comparison, the software side: the 31-node ESN (paper's baseline, one reservoir, fresh-run protocol) with its
input u quantized to N bits over [0, 1] and no other noise.

The simulated line powers are saved (``--out``.npz) so the quantization can be redone without re-simulating.

    python experiments/readout_quantization.py --device cpu
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
from combrc import make_task, to_features  # noqa: E402
from combrc.evaluate import WARMUP, held_out  # noqa: E402
from combrc.readout import add_taps, detector_floor, measure  # noqa: E402
from reference_ring import jitter_process, runs  # noqa: E402

LINES, TAPS = 31, 8
BITS = [4, 5, 6, 7, 8, 9, 10, 12, 14]


def quantize(x, lo, hi, bits):
    """Uniform N-bit quantization of x over [lo, hi] (per column), clipped to the range."""
    levels = 2 ** bits - 1
    q = torch.round((x - lo) / (hi - lo) * levels).clamp(0, levels)
    return lo + q * (hi - lo) / levels


def readings(lines, cfg, gen):
    """Noisy photodetector readings (W) of one run [T, LINES], as in the paper's readout."""
    detector, shot_time = ce.detection(cfg, "realistic")
    return measure(lines, detector, gen, shot_time=shot_time)


def features(p, cal, bits, mode, floor):
    """Readout features of readings ``p`` digitised with an ADC calibrated on readings ``cal``."""
    if bits is None:
        return add_taps(to_features(p, floor=floor), TAPS)
    if mode == "linear":
        hi = 1.1 * cal.max(0).values.clamp_min(floor)
        return add_taps(to_features(quantize(p.clamp_min(0), torch.zeros_like(hi), hi, bits), floor=floor), TAPS)
    xc, x = to_features(cal, floor=floor), to_features(p, floor=floor)
    lo, hi = xc.min(0).values, xc.max(0).values
    span = (hi - lo).clamp_min(1e-6)
    return add_taps(quantize(x, lo - 0.05 * span, hi + 0.05 * span, bits), TAPS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="mask_drop")
    ap.add_argument("--trial", type=int, default=211)
    ap.add_argument("--series-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--n-modes", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/readout_quantization")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cfg = ce.sample(args.group, args.trial)
    corr = ce.REALISTIC.get("detuning_corr", 1.0)
    floor = detector_floor(ce.detection(cfg, "realistic")[0])
    print(f"{args.group} trial {args.trial}: {cfg['p_total'] * 1e3:.0f} mW; ADC bits {BITS}", flush=True)
    variants = [("none", None)] + [(m, b) for m in ("linear", "log") for b in BITS]
    ring = {v: [] for v in variants}
    esn = {b: [] for b in [None] + BITS}
    saved = {}
    for seed in args.series_seeds:
        u_np, y_np = make_task("mackey_glass_h10", WARMUP + ce.N_TOTAL, seed=seed)
        u = torch.tensor(u_np, dtype=torch.float64)[: WARMUP + ce.N_TOTAL]
        y = torch.tensor(y_np, dtype=torch.float64)[WARMUP: WARMUP + ce.N_TOTAL]
        cache = f"{args.out}_lines_s{seed}.npy"
        if os.path.exists(cache):
            lines = torch.tensor(np.load(cache))
        else:
            drift = jitter_process(2 * args.pairs, len(u), corr, seed=17 + seed)
            lines = runs(cfg, u, drift, args.n_modes, args.steps, args.device, seed)
            np.save(cache, lines.numpy())
        saved[seed] = cache
        for v in variants:
            tests = []
            for k in range(args.pairs):
                gen = torch.Generator().manual_seed(1000 + k)
                pa, pb = readings(lines[2 * k], cfg, gen), readings(lines[2 * k + 1], cfg, gen)
                xa = features(pa, pa, v[1], v[0], floor)
                xb = features(pb, pa, v[1], v[0], floor)      # the test run uses the calibration of the fitting run
                tests.append(held_out(xa, y, x_test=xb, n_test=ce.N_TEST)[1])
            ring[v].append(float(np.mean(tests)))
        for b in esn:
            uq = u if b is None else quantize(u, 0.0, 1.0, b)
            esn[b].append(ce.esn_fresh(uq, torch.tensor(y_np, dtype=torch.float64), LINES, TAPS, 0.0, 1)[0])
        print(f"series {seed}: ring none {ring[variants[0]][-1]:.2e}; "
              + "  ".join(f"{m[:3]}{b} {ring[(m, b)][-1]:.2e}" for m, b in variants[1:] if b in (6, 8, 10))
              + f"; ESN31 8 bit {esn[8][-1]:.2e}", flush=True)
    out = []
    for m, b in variants:
        out.append({"system": "ring", "adc": m, "bits": "" if b is None else b,
                    "test_nmse": ce.gmean(ring[(m, b)]), "per_series": " ".join(f"{x:.3e}" for x in ring[(m, b)])})
    for b in esn:
        out.append({"system": "ESN 31", "adc": "input" if b else "none", "bits": "" if b is None else b,
                    "test_nmse": ce.gmean(esn[b]), "per_series": " ".join(f"{x:.3e}" for x in esn[b])})
    for r in out:
        print(f"{r['system']:7s} {r['adc']:7s} {str(r['bits']):>3s} bits  {r['test_nmse']:.2e}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)


if __name__ == "__main__":
    main()
