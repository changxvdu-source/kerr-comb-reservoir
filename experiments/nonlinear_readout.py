"""Feasibility test: can a nonlinear readout remove the detuning-jitter floor that a linear readout cannot?

The paper finds that, at the best single ring on noisy Mackey-Glass, about half of the error floor is detuning
jitter, and that a linear readout cannot remove it even when handed the true jitter sequence
(reference_ring.py). This script asks whether a readout that is nonlinear in the detected line features, and
that is NOT told the jitter, can do better. Operating point and noise as in reference_ring.py (mask_drop trial
211, realistic noise). Per series, ``--runs`` runs with independent jitter and noise are simulated, plus two
runs with the detuning jitter switched off (all other noise on) for the jitter-free reference.

Readouts (fitted on the first K runs, ridge strength / early stopping chosen on the validation part of those
runs, tested on the test part of the last run, which has independent noise; K = 1 is the paper's cross-run
protocol, K > 1 lets a flexible readout see several jitter realisations):
  linear          ridge on log line powers, 31 lines x 8 taps (the paper's readout)
  quadratic       linear + all products of the current 31 line features (496 more)
  mlp             (--mlp) small two-layer network (tanh) on the standardised 31 x 8 features, plus a linear skip path
Each readout is applied to three sources: the ring, the same ring with the detuning jitter switched off (how
far removing the jitter could go) and the paper's 31-node ESN with input noise 3e-3 (fairness: software gets the
same readout).

    python experiments/nonlinear_readout.py --device cuda
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
import reference_ring as rr  # noqa: E402
from combrc import make_task  # noqa: E402
from combrc.evaluate import WARMUP, esn_states  # noqa: E402
from combrc.readout import add_taps, fit_readout, nmse  # noqa: E402

LINES = rr.LINES


def quad(x):
    cur = x[:, :LINES]
    i, j = torch.triu_indices(LINES, LINES)
    return torch.cat([x, cur[:, i] * cur[:, j]], 1)


def ridge(xs_fit, ys_fit, n_train, n_val, x_test, y_test):
    """Ridge over the concatenated training parts of the fitting runs; ridge strength on their validation parts."""
    xtr = torch.cat([x[:n_train] for x in xs_fit]); ytr = torch.cat([y[:n_train] for y in ys_fit])
    xva = torch.cat([x[n_train:n_train + n_val] for x in xs_fit]); yva = torch.cat([y[n_train:n_train + n_val] for y in ys_fit])
    model = fit_readout(torch.cat([xtr, xva]), torch.cat([ytr, yva]), len(xtr), len(xva))
    return nmse(y_test, model.predict(x_test)).item()


def mlp(xs_fit, ys_fit, n_train, n_val, x_test, y_test, hidden=64, seed=0, epochs=3000, wd=1e-4):
    torch.manual_seed(seed)
    xtr = torch.cat([x[:n_train] for x in xs_fit]).float(); ytr = torch.cat([y[:n_train] for y in ys_fit]).float()
    xva = torch.cat([x[n_train:n_train + n_val] for x in xs_fit]).float(); yva = torch.cat([y[n_train:n_train + n_val] for y in ys_fit]).float()
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-8
    ym, ys = ytr.mean(), ytr.std()
    norm = lambda a: (a - mu) / sd  # noqa: E731
    d = xtr.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, hidden),
                              torch.nn.Tanh(), torch.nn.Linear(hidden, 1))
    skip = torch.nn.Linear(d, 1)
    params = list(net.parameters()) + list(skip.parameters())
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=wd)
    f = lambda a: (net(norm(a)) + skip(norm(a))).squeeze(-1) * ys + ym  # noqa: E731
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        opt.zero_grad()
        loss = ((f(xtr) - ytr) ** 2).mean()
        loss.backward()
        opt.step()
        if ep % 20 == 0:
            with torch.no_grad():
                v = nmse(yva.double(), f(xva).double()).item()
            if v < best:
                best, bad = v, 0
                best_state = [p.detach().clone() for p in params]
            else:
                bad += 1
                if bad > 15:
                    break
    with torch.no_grad():
        for p, s in zip(params, best_state):
            p.copy_(s)
        return nmse(y_test, f(x_test.float()).double()).item()


def esn_runs(u, y, runs, input_noise, n_train, n_val, size=LINES, taps=8):
    """31-node ESN features (8 taps) of ``runs`` realisations of the input noise; hyper-parameters chosen as in
    the paper (validation error of the linear readout on the first realisation)."""
    target = y

    def states(radius, scale, k):
        gen = torch.Generator().manual_seed(100 + k)
        seen = u + input_noise * torch.randn(u.shape, generator=gen, dtype=u.dtype) if input_noise else u
        return add_taps(esn_states(seen, size, radius, scale), taps)[WARMUP:]

    def val(radius, scale):
        x = states(radius, scale, 0)
        m = fit_readout(x, target, n_train, n_val)
        return nmse(target[n_train:n_train + n_val], m.predict(x[n_train:n_train + n_val])).item()

    _, radius, scale = min((val(r, s), r, s) for r in (0.5, 0.8, 0.95, 1.1) for s in (0.1, 0.3, 1.0))
    return [states(radius, scale, k) for k in range(runs)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="mask_drop")
    ap.add_argument("--trial", type=int, default=211)
    ap.add_argument("--series-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--runs", type=int, default=5, help="runs with jitter per series (the last one is the test run)")
    ap.add_argument("--fit-runs", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--symbols", type=int, default=ce.N_TOTAL, help="scored symbols per run (paper: 1700)")
    ap.add_argument("--esn-noise", type=float, default=3e-3, help="ESN input noise (paper: 3e-3)")
    ap.add_argument("--mlp", action="store_true", help="also the small neural network (data-starved at 1700 symbols)")
    ap.add_argument("--n-modes", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/nonlinear_readout")
    args = ap.parse_args()
    torch.set_num_threads(4)
    cfg = ce.sample(args.group, args.trial)
    corr = ce.REALISTIC.get("detuning_corr", 1.0)
    n = args.symbols
    n_train, n_val = int(0.6 * n), int(0.2 * n)
    print(f"{args.group} trial {args.trial}: {cfg['p_total'] * 1e3:.0f} mW; {n} symbols per run, "
          f"{args.runs} jittered runs + 2 jitter-free per series", flush=True)
    sources = ["ring", "ring, no jitter", "ESN"]
    readouts = ["linear", "quadratic"] + (["mlp"] if args.mlp else [])
    names = [f"{src}: {r} K={k}" for src in sources for k in args.fit_runs for r in readouts]
    per = {v: [] for v in names}
    for seed in args.series_seeds:
        u_np, y_np = make_task("mackey_glass_h10", WARMUP + n, seed=seed)
        u = torch.tensor(u_np, dtype=torch.float64)[: WARMUP + n]
        y = torch.tensor(y_np, dtype=torch.float64)[WARMUP: WARMUP + n]
        drift = rr.jitter_process(args.runs, len(u), corr, seed=17 + seed)
        lines = rr.runs(cfg, u, drift, args.n_modes, args.steps, args.device, seed)
        lines0 = rr.runs(cfg, u, torch.zeros_like(drift), args.n_modes, args.steps, args.device, seed + 50)
        gen = torch.Generator().manual_seed(1000 + seed)
        feats = {"ring": [rr.features(lines[r], None, cfg, gen, "lines") for r in range(args.runs)],
                 "ring, no jitter": [rr.features(lines0[r], None, cfg, gen, "lines") for r in range(args.runs)],
                 "ESN": esn_runs(u, y, args.runs, args.esn_noise, n_train, n_val)}
        sl = slice(n_train + n_val, n)
        yt = y[sl]
        for src in sources:
            test_x = feats[src][-1]
            for k in args.fit_runs:
                fit = feats[src][:k]
                ys = [y] * k
                per[f"{src}: linear K={k}"].append(ridge(fit, ys, n_train, n_val, test_x[sl], yt))
                per[f"{src}: quadratic K={k}"].append(
                    ridge([quad(x) for x in fit], ys, n_train, n_val, quad(test_x)[sl], yt))
                if args.mlp:
                    per[f"{src}: mlp K={k}"].append(float(np.mean(
                        [mlp(fit, ys, n_train, n_val, test_x[sl], yt, seed=q) for q in range(3)])))
        print(f"series {seed}: " + "   ".join(f"{v} {per[v][-1]:.2e}" for v in names), flush=True)
    base = ce.gmean(per[f"ring: linear K={args.fit_runs[0]}"])
    rows = []
    for v in names:
        g = ce.gmean(per[v])
        rows.append({"readout": v, "test_nmse": g, "vs_linear_K1": base / g,
                     "per_series": " ".join(f"{x:.3e}" for x in per[v])})
        print(f"{v:30s} {g:.2e}   ({base / g:.2f}x better than the ring's linear readout, K={args.fit_runs[0]})")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved", args.out + ".csv")


if __name__ == "__main__":
    main()
