"""The reservoir under a realistic pump laser: detuning jitter from a laser frequency-noise spectrum.

The paper's noise model gives the ring a per-symbol detuning jitter of 1 % of the half-linewidth. Here the
jitter is instead the frequency noise of the pump laser relative to the ring, with the standard spectrum
    S(f) = h0 + h_{-1} / f      (one-sided, Hz^2/Hz),   h0 = linewidth / pi,   h_{-1} = h0 * corner,
i.e. white frequency noise of a Lorentzian linewidth plus flicker noise below the ``--corner`` frequency
(about 1 MHz for semiconductor lasers). The laser is held on the resonance by a lock of bandwidth
``--lock-bw`` (a first-order loop: the residual spectrum is S(f) f^2 / (f^2 + B^2)); without a lock the record
starts at 1/(``--record`` x symbol time). The time series is synthesised from the spectrum, averaged over
each symbol and divided by the ring's half-linewidth; that sequence is the detuning jitter of each run.
Everything else is the paper's realistic noise.

Operating point: the best single ring on noisy Mackey-Glass (stronger drop coupling, trial 211, Q 1.9e6,
1 ns symbols, 1.55e-4 on new runs with the 1 % assumption). Readout: 31 lines x 8 taps, fitted on one run of
a pair and tested on the other, ``--pairs`` pairs per series.

    python experiments/laser_noise.py --linewidth 1.6e3                       # white only: checks the mapping
    python experiments/laser_noise.py --linewidth 1e5 --corner 1e6 --lock-bw 1e6
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
from combrc import make_task  # noqa: E402
from combrc.evaluate import WARMUP, held_out  # noqa: E402

NU = 299_792_458 / 1550e-9


def laser_frequency(rows, n, T, linewidth, corner, lock_bw, record, seed, oversample=8):
    """[rows, n] laser frequency offsets in Hz, each the mean over one symbol of length T, synthesised from
    the one-sided spectrum h0 (1 + corner / f) with the lock's high-pass; a random window of ``n`` symbols
    out of a record of ``record`` symbols (so slow components appear as drifts and offsets)."""
    rng = np.random.default_rng(seed)
    N = record * oversample
    dt = T / oversample
    f = np.fft.rfftfreq(N, dt)
    h0 = linewidth / math.pi
    S = np.zeros_like(f)
    S[1:] = h0 * (1 + corner / f[1:])
    if lock_bw:
        S *= f ** 2 / (f ** 2 + lock_bw ** 2)
    amp = np.sqrt(S * N / (2 * dt))
    out = np.empty((rows, n))
    for r in range(rows):
        c = amp * (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / math.sqrt(2)
        x = np.fft.irfft(c, n=N)
        start = rng.integers(0, record - n) * oversample
        out[r] = x[start: start + n * oversample].reshape(n, oversample).mean(1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--linewidth", type=float, required=True, help="Lorentzian (white-noise) linewidth, Hz")
    ap.add_argument("--corner", type=float, default=0.0, help="flicker corner frequency, Hz (0: white only)")
    ap.add_argument("--lock-bw", type=float, default=0.0, help="bandwidth of the laser-to-ring lock, Hz (0: none)")
    ap.add_argument("--record", type=int, default=2 ** 17, help="synthesised record length in symbols")
    ap.add_argument("--group", default="mask_drop")
    ap.add_argument("--trial", type=int, default=211)
    ap.add_argument("--series-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--n-modes", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--check", action="store_true", help="only check the noise statistics and exit")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(4)
    cfg = ce.sample(args.group, args.trial)
    ring = ce.build(cfg, args.n_modes, args.steps)
    T = cfg.get("symbol_time", 1e-9)
    hwhm = NU / (2 * float(ring.q_total))
    n = WARMUP + ce.N_TOTAL
    if args.check:
        nu = laser_frequency(64, n, T, args.linewidth, args.corner, args.lock_bw, args.record, seed=1)
        expect = math.sqrt(args.linewidth / (2 * math.pi * T))
        step = np.diff(nu, axis=1).std() / math.sqrt(2)
        print(f"Q {float(ring.q_total):.2e}, half-linewidth {hwhm / 1e6:.1f} MHz")
        print(f"rms per-symbol frequency {nu.std():.3e} Hz (white-only expectation {expect:.3e}); "
              f"symbol-to-symbol part {step:.3e}; rms within a run {nu.std(1).mean():.3e}, "
              f"spread of run means {nu.mean(1).std():.3e}")
        print(f"jitter in half-linewidths: rms {nu.std() / hwhm:.2%}")
        return
    ce.REALISTIC["detuning"] = 1.0              # the supplied sequence is the jitter in half-linewidths
    tests, eps_rms = [], []
    for seed in args.series_seeds:
        u_np, y_np = make_task("mackey_glass_h10", n, seed=seed)
        u = torch.tensor(u_np, dtype=torch.float64)[:n]
        y = torch.tensor(y_np, dtype=torch.float64)[WARMUP: n]
        rows = 2 * args.pairs
        eps = laser_frequency(rows, n, T, args.linewidth, args.corner, args.lock_bw, args.record,
                              seed=1000 + seed) / hwhm
        eps_rms.append(float(eps.std()))
        lines = rr.runs(cfg, u, torch.tensor(eps), args.n_modes, args.steps, args.device, seed)
        d = torch.tensor(eps)[:, WARMUP:]
        per = []
        for k in range(args.pairs):
            gen = torch.Generator().manual_seed(1000 + k)
            xa = rr.features(lines[2 * k], d[2 * k], cfg, gen, "lines")
            xb = rr.features(lines[2 * k + 1], d[2 * k + 1], cfg, gen, "lines")
            per.append(held_out(xa, y, x_test=xb, n_test=ce.N_TEST)[1])
        tests.append(float(np.mean(per)))
        print(f"series {seed}: jitter rms {eps_rms[-1]:.2%} of the half-linewidth, test NMSE {tests[-1]:.2e}", flush=True)
    row = {"linewidth_hz": args.linewidth, "corner_hz": args.corner, "lock_bw_hz": args.lock_bw,
           "jitter_rms": float(np.mean(eps_rms)), "test_nmse": ce.gmean(tests),
           "per_series": " ".join(f"{t:.3e}" for t in tests)}
    print(f"linewidth {args.linewidth:g} Hz, corner {args.corner:g} Hz, lock {args.lock_bw:g} Hz: "
          f"jitter {row['jitter_rms']:.2%}, test NMSE {row['test_nmse']:.2e}")
    out = args.out or f"results/laser_noise_lw{args.linewidth:g}_c{args.corner:g}_l{args.lock_bw:g}"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
