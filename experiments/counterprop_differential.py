"""Idea B, first feasibility test: counter-propagating modes of one Kerr ring with a differential readout.

Both directions of one ring are pumped from the same laser, so laser frequency jitter (a detuning jitter) and
laser intensity noise are common to them, while the signal can be put on the difference. The two directions are
coupled by cross-phase modulation (factor 2), which gives the known spontaneous symmetry breaking; near it the
response to asymmetric drives is amplified. Question: does pumping both directions, with the signal on their
difference, lower the jitter penalty and the error compared with the same ring pumped in one direction?

Single-mode model (no comb), normalised: time in units of 2/kappa (field decay time), detuning in half-linewidths,
power in units of the Kerr shift:
    dA/dt = -(1 + i D(t)) A + i (|A|^2 + 2 |B|^2) A + F_A(t)
    dB/dt = -(1 + i D(t)) B + i (|B|^2 + 2 |A|^2) B + F_B(t)
D(t) = D0 + j d(t): common detuning jitter, d an AR(1) process with correlation 1/e per symbol and unit
variance, j = 0.01 (1 % of the half-linewidth, as in the paper). F_X = F (1 + r(t)) (1 + s_X m (u - 1/2)):
common intensity noise r (1e-3 rms per symbol); s = (+1, 0) "uni" (B not pumped), (+1, +1) "bi-common",
(+1, -1) "bi-diff", (+1, 0) with B pumped at F "bi-one", and "two-rings-diff": the drives of "bi-diff" on two
separate rings fed from the same laser (common jitter and intensity noise, no cross-phase coupling), the fair
control for the two detected channels. Independent field noise in each direction and
relative detection noise on each detected power (``--field``, ``--detect``). One symbol lasts ``--ts`` time
units (0.3 ~ Q 2e6 at 1 GBd). Readout: log detected powers of the pumped directions with ``--taps`` taps
(linear ridge, and with all products of the current features), fitted on one run and tested on another run with
independent noise; Mackey-Glass, 10 steps ahead.

    python experiments/counterprop_differential.py
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
from combrc import make_task  # noqa: E402
from combrc.evaluate import WARMUP  # noqa: E402
from combrc.readout import add_taps, fit_readout, nmse  # noqa: E402

MODES = {"uni": (1.0, 0.0, False, 1.0), "bi-common": (1.0, 1.0, True, 1.0), "bi-diff": (1.0, -1.0, True, 1.0),
         "bi-one": (1.0, 0.0, True, 1.0), "two-rings-diff": (1.0, -1.0, True, 0.0)}


def ar1(rows, T, gen, corr=1.0):
    rho = math.exp(-1.0 / corr)
    d = torch.empty(rows, T, dtype=torch.float64)
    d[:, 0] = torch.randn(rows, generator=gen, dtype=torch.float64)
    e = torch.randn(rows, T, generator=gen, dtype=torch.float64) * math.sqrt(1 - rho ** 2)
    for t in range(1, T):
        d[:, t] = rho * d[:, t - 1] + e[:, t]
    return d


def simulate(u, D0, F, sA, sB, pumpB, m, ts, jitter, rin, field, detect, seed, xpm, dt=0.02):
    """Detected powers [rows, T, 2] of rows = len(D0) independent runs (one per configuration row)."""
    rows, T = len(D0), len(u)
    gen = torch.Generator().manual_seed(seed)
    d = ar1(rows, T, gen) * jitter
    r = torch.randn(rows, T, generator=gen, dtype=torch.float64) * rin
    c = (u - 0.5)[None, :]
    FA = (F[:, None] * (1 + r) * (1 + sA[:, None] * m[:, None] * c)).to(torch.complex128)
    FB = (F[:, None] * (1 + r) * (1 + sB[:, None] * m[:, None] * c) * pumpB[:, None]).to(torch.complex128)
    D = D0[:, None] + d
    n = max(1, round(ts / dt))
    h = ts / n
    A = (torch.randn(rows, generator=gen, dtype=torch.float64) * 1e-3).to(torch.complex128)
    B = (torch.randn(rows, generator=gen, dtype=torch.float64) * 1e-3).to(torch.complex128)
    out = torch.empty(rows, T, 2, dtype=torch.float64)
    sq = field * math.sqrt(h)
    for t in range(T):
        fa, fb, dd = FA[:, t], FB[:, t], D[:, t]

        def rhs(a, b):
            pa, pb = a.real ** 2 + a.imag ** 2, b.real ** 2 + b.imag ** 2
            return (-(1 + 1j * dd) * a + 1j * (pa + 2 * xpm * pb) * a + fa,
                    -(1 + 1j * dd) * b + 1j * (pb + 2 * xpm * pa) * b + fb)

        ia = torch.zeros(rows, dtype=torch.float64)
        ib = torch.zeros(rows, dtype=torch.float64)
        for _ in range(n):
            k1 = rhs(A, B)
            k2 = rhs(A + 0.5 * h * k1[0], B + 0.5 * h * k1[1])
            k3 = rhs(A + 0.5 * h * k2[0], B + 0.5 * h * k2[1])
            k4 = rhs(A + h * k3[0], B + h * k3[1])
            A = A + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            B = B + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            if sq:
                A = A + sq * torch.complex(*torch.randn(2, rows, generator=gen, dtype=torch.float64))
                B = B + sq * torch.complex(*torch.randn(2, rows, generator=gen, dtype=torch.float64))
            ia += A.abs() ** 2
            ib += B.abs() ** 2
        out[:, t, 0], out[:, t, 1] = ia / n, ib / n
    noise = 1 + detect * torch.randn(rows, T, 2, generator=gen, dtype=torch.float64)
    return out * noise


def quad(x, k):
    cur = x[:, :k]
    i, j = torch.triu_indices(k, k)
    return torch.cat([x, cur[:, i] * cur[:, j]], 1)


def score(pa, pb, y, two, taps, kind):
    """Cross-run test NMSE: fitted on run a, tested on run b. pa/pb: [T, 2] detected powers after warm-up."""
    k = 2 if two else 1
    xa, xb = (torch.log(p[:, :k].clamp_min(1e-9)) for p in (pa, pb))
    xa, xb = add_taps(xa, taps), add_taps(xb, taps)
    if kind == "quad":
        xa, xb = quad(xa, k), quad(xb, k)
    if not (torch.isfinite(xa).all() and torch.isfinite(xb).all()):
        return float("inf")
    n = len(y)
    n_train, n_val = int(0.6 * n), int(0.2 * n)
    model = fit_readout(xa, y, n_train, n_val)
    return nmse(y[n_train + n_val:], model.predict(xb)[n_train + n_val:]).item()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detunings", type=float, nargs=3, default=[-1.0, 6.0, 15], metavar=("FROM", "TO", "N"))
    ap.add_argument("--powers", type=float, nargs="+", default=[0.5, 1, 2, 3, 4, 6, 8, 12], help="F^2 values")
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.3])
    ap.add_argument("--ts", type=float, default=0.3)
    ap.add_argument("--taps", type=int, default=32)
    ap.add_argument("--jitter", type=float, default=0.01)
    ap.add_argument("--rin", type=float, default=1e-3)
    ap.add_argument("--field", type=float, default=1e-3)
    ap.add_argument("--detect", type=float, default=1e-3)
    ap.add_argument("--series-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--symbols", type=int, default=1700)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/counterprop_differential")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dets = np.linspace(args.detunings[0], args.detunings[1], int(args.detunings[2]))
    grid = [(mode, D, P, m) for mode in MODES for D in dets for P in args.powers for m in args.depths]
    print(f"{len(grid)} configurations, ts {args.ts}, taps {args.taps}", flush=True)
    D0 = torch.tensor([g[1] for g in grid], dtype=torch.float64)
    F = torch.tensor([math.sqrt(g[2]) for g in grid], dtype=torch.float64)
    sA = torch.tensor([MODES[g[0]][0] for g in grid], dtype=torch.float64)
    sB = torch.tensor([MODES[g[0]][1] for g in grid], dtype=torch.float64)
    pumpB = torch.tensor([1.0 if MODES[g[0]][2] else 0.0 for g in grid], dtype=torch.float64)
    m = torch.tensor([g[3] for g in grid], dtype=torch.float64)
    xpm = torch.tensor([MODES[g[0]][3] for g in grid], dtype=torch.float64)
    res = {}
    for seed in args.series_seeds:
        u_np, y_np = make_task("mackey_glass_h10", WARMUP + args.symbols, seed=seed)
        u = torch.tensor(u_np, dtype=torch.float64)[: WARMUP + args.symbols]
        y = torch.tensor(y_np, dtype=torch.float64)[WARMUP: WARMUP + args.symbols]
        for label, jit in (("jitter", args.jitter), ("no jitter", 0.0)):
            runs = [simulate(u, D0, F, sA, sB, pumpB, m, args.ts, jit, args.rin, args.field, args.detect,
                             seed=10_000 * seed + 100 * (jit > 0) + r, xpm=xpm)[:, WARMUP:] for r in range(2)]
            for i, g in enumerate(grid):
                two = MODES[g[0]][2]
                for kind in ("linear", "quad"):
                    res.setdefault((i, label, kind), []).append(score(runs[0][i], runs[1][i], y, two, args.taps, kind))
            print(f"series {seed} {label} done", flush=True)
    gm = lambda v: float(np.exp(np.mean(np.log(np.maximum(v, 1e-30)))))  # noqa: E731
    rows = []
    for i, (mode, D, P, mm) in enumerate(grid):
        row = {"mode": mode, "detuning": D, "F2": P, "depth": mm}
        for kind in ("linear", "quad"):
            a, b = gm(res[(i, "jitter", kind)]), gm(res[(i, "no jitter", kind)])
            row[f"{kind}_jitter"], row[f"{kind}_no_jitter"], row[f"{kind}_penalty"] = a, b, a / b
        rows.append(row)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for kind in ("linear", "quad"):
        print(f"\nbest per mode ({kind} readout, with jitter):")
        for mode in MODES:
            mine = sorted([r for r in rows if r["mode"] == mode], key=lambda r: r[f"{kind}_jitter"])[:3]
            for r in mine:
                print(f"  {mode:10s} D {r['detuning']:5.2f} F2 {r['F2']:4.1f} m {r['depth']:.1f}: "
                      f"{r[f'{kind}_jitter']:.2e} (no jitter {r[f'{kind}_no_jitter']:.2e}, penalty {r[f'{kind}_penalty']:.2f})")
    print("saved", args.out + ".csv")


if __name__ == "__main__":
    main()
