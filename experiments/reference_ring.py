"""How much of the error floor could a reference ring remove? The ideal-reference bound.

Idea (shared-reference ring array): rings on one chip fed by one laser see the same laser intensity noise
and the same laser frequency jitter (which acts as detuning jitter); a reference ring without signal would
measure that common-mode noise, and the readout could subtract it. Before simulating a real reference ring,
this measures the upper bound: the readout is handed the true detuning-jitter sequence of each run.

Operating point: the best single ring on noisy Mackey-Glass (stronger drop coupling, trial 211 of
floor_mg_pin_trials.csv, 1.55e-4 on new runs). Realistic noise (intensity noise, 1 % detuning jitter,
quantum field noise, shot noise, 0.3 uW detectors); the jitter sequence of every run is drawn here and
passed to the simulator. ``--pairs`` pairs of new runs per series; the readout is fitted on one run of a
pair and tested on the other (cross-run, as in the paper). Readouts, all ridge over 31 lines x 8 taps plus:
  lines             nothing else (the paper's readout)
  + jitter          the jitter of the last 8 symbols (what a reference ring and a linear readout could use at best)
  + jitter x lines  also the products of the current jitter with the current line features (a readout
                    that can scale its correction with the ring's state)
and the same ring with the detuning jitter switched off (all other noise on): how far removing it could go.

    python experiments/reference_ring.py --device cuda
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
from combrc import make_task, select_lines, to_features  # noqa: E402
from combrc.evaluate import WARMUP, held_out  # noqa: E402
from combrc.lle import stack_rings  # noqa: E402
from combrc.readout import add_taps, detector_floor, measure  # noqa: E402

LINES, TAPS = 31, 8


def jitter_process(rows, T, corr, seed):
    """Unit-variance AR(1) detuning drift with correlation time ``corr`` symbols, as the simulator draws it."""
    rho = math.exp(-1.0 / corr) if corr > 0 else 0.0
    gen = torch.Generator().manual_seed(seed)
    d = torch.empty(rows, T, dtype=torch.float64)
    d[:, 0] = torch.randn(rows, generator=gen, dtype=torch.float64)
    for t in range(1, T):
        d[:, t] = rho * d[:, t - 1] + math.sqrt(1 - rho ** 2) * torch.randn(rows, generator=gen, dtype=torch.float64)
    return d


@torch.no_grad()
def runs(cfg, u, drift, n_modes, steps, device, seed, block=200):
    """Central comb lines [rows, symbols after WARMUP, LINES] of ``len(drift)`` runs with the given jitter."""
    rows = len(drift)
    ring = ce.build(cfg, n_modes, steps)
    stack = ce.with_noise(stack_rings([ring], repeats=rows).to(device), "realistic")
    state = torch.cat([ring.init_state(1, seed=100 + s) for s in range(rows)]).to(device)
    gen = torch.Generator(device=device).manual_seed(4_000_037 + 7919 * seed)
    kept = []
    for t0 in range(0, len(u), block):
        spectra, state = stack(u[t0: t0 + block].to(device).expand(rows, -1), state=state, generator=gen,
                               detuning=drift[:, t0: t0 + block].to(device))
        if t0 + block > WARMUP:
            kept.append(select_lines(spectra[:, max(WARMUP - t0, 0):], LINES).cpu())
    return torch.cat(kept, 1)


def features(lines, drift, cfg, gen, extra):
    detector, shot_time = ce.detection(cfg, "realistic")
    x = add_taps(to_features(measure(lines, detector, gen, shot_time=shot_time), floor=detector_floor(detector)), TAPS)
    if extra == "lines":
        return x
    j = drift[:, None].clone()
    parts = [x, add_taps(j, TAPS)]
    if extra == "jitter x lines":
        parts.append(j * x[:, :LINES])
    return torch.cat(parts, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="mask_drop")
    ap.add_argument("--trial", type=int, default=211)
    ap.add_argument("--series-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--n-modes", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/reference_ring_ideal")
    args = ap.parse_args()
    torch.set_num_threads(4)
    cfg = ce.sample(args.group, args.trial)
    corr = ce.REALISTIC.get("detuning_corr", 1.0)
    print(f"{args.group} trial {args.trial}: {cfg['p_total'] * 1e3:.0f} mW, jitter {ce.REALISTIC['detuning']:.1%} "
          f"of the half-linewidth, correlation {corr} symbol(s)", flush=True)
    variants = ["lines", "+ jitter", "+ jitter x lines", "no jitter"]
    per = {v: [] for v in variants}
    for seed in args.series_seeds:
        u_np, y_np = make_task("mackey_glass_h10", WARMUP + ce.N_TOTAL, seed=seed)
        u = torch.tensor(u_np, dtype=torch.float64)[: WARMUP + ce.N_TOTAL]
        y = torch.tensor(y_np, dtype=torch.float64)[WARMUP: WARMUP + ce.N_TOTAL]
        rows = 2 * args.pairs
        drift = jitter_process(rows, len(u), corr, seed=17 + seed)
        lines = runs(cfg, u, drift, args.n_modes, args.steps, args.device, seed)
        lines0 = runs(cfg, u, torch.zeros_like(drift), args.n_modes, args.steps, args.device, seed)
        d = drift[:, WARMUP:]
        for v in variants:
            tests = []
            for k in range(args.pairs):
                gen = torch.Generator().manual_seed(1000 + k)
                src = lines0 if v == "no jitter" else lines
                extra = "lines" if v == "no jitter" else {"lines": "lines", "+ jitter": "jitter",
                                                          "+ jitter x lines": "jitter x lines"}[v]
                xa = features(src[2 * k], d[2 * k], cfg, gen, extra)
                xb = features(src[2 * k + 1], d[2 * k + 1], cfg, gen, extra)
                tests.append(held_out(xa, y, x_test=xb, n_test=ce.N_TEST)[1])
            per[v].append(float(np.mean(tests)))
        print(f"series {seed}: " + "   ".join(f"{v} {per[v][-1]:.2e}" for v in variants), flush=True)
    rows_out = [{"readout": v, "test_nmse": ce.gmean(per[v]), "per_series": " ".join(f"{x:.3e}" for x in per[v])}
                for v in variants]
    base = rows_out[0]["test_nmse"]
    for r in rows_out:
        print(f"{r['readout']:18s} {r['test_nmse']:.2e}   ({base / r['test_nmse']:.2f}x better than the plain readout)")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)


if __name__ == "__main__":
    main()
