"""Fairness check for the mask optimisation: give the ESN's input weights the same optimisation budget.

The ring's best result uses a mask optimised by random search + evolution (mask_optimize.py). Here the ESN's input
weights and biases (its counterpart of the mask; the recurrent matrix stays fixed, as the ring's physics does) get the
same procedure and budget: ``--population`` random candidates on series 0, then ``--generations`` generations of
``--children`` children around the ``--elite`` best, sigma shrinking by ``--decay``; choice of the ``--final`` best on
3 pairs of noise realisations of series 0-2; report on held-out series 3-5 (3 pairs). A "run" is one realisation of
the input noise; readouts are fitted on one run and tested on another, 8 taps, as everywhere in the paper.
Spectral radius and input scale are first chosen by the paper's grid (esn_fresh) with the random input weights.

    python experiments/esn_optimize.py --size 62 --noise 1e-3
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import compare_encodings as ce  # noqa: E402
import counterprop_comb as cc  # noqa: E402
from combrc.evaluate import WARMUP, held_out  # noqa: E402
from combrc.readout import add_taps  # noqa: E402

TAPS = 8


def reservoir(n, radius, seed=0):
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(n, n, generator=gen, dtype=torch.float64)
    w = w * (radius / torch.linalg.eigvals(w).abs().max())
    w_in = 2 * torch.rand(n, generator=gen, dtype=torch.float64) - 1     # as combrc.evaluate.esn_states
    bias = 0.2 * (2 * torch.rand(n, generator=gen, dtype=torch.float64) - 1)
    return w, w_in, bias


def states(w, W_in, Bias, U):
    """Batched ESN states. W_in, Bias [P, n]; U [R, T] noisy inputs -> [P, R, T, n]."""
    P, n = W_in.shape
    R, T = U.shape
    x = torch.zeros(P, R, n, dtype=torch.float64)
    v = 2 * U - 1
    out = torch.empty(P, R, T, n, dtype=torch.float64)
    wt = w.T.contiguous()
    for t in range(T):
        x = torch.tanh(x @ wt + W_in[:, None, :] * v[None, :, t, None] + Bias[:, None, :])
        out[:, :, t] = x
    return out


def score(w, cands, seeds, pairs, noise, salt):
    """Mean validation / test NMSE per candidate (geometric mean over series), cross-run."""
    W_in = torch.stack([c[0] for c in cands])
    Bias = torch.stack([c[1] for c in cands])
    vals, tests = np.zeros((len(cands), len(seeds))), np.zeros((len(cands), len(seeds)))
    for k, s in enumerate(seeds):
        u, y = cc.task(s)
        gen = torch.Generator().manual_seed(77_000 + 1000 * salt + s)
        U = u[None, :] + noise * torch.randn(2 * pairs, len(u), generator=gen, dtype=torch.float64)
        for p0 in range(0, len(cands), 64):               # chunks keep the state tensor small
          S = states(w, W_in[p0: p0 + 64], Bias[p0: p0 + 64], U)[:, :, WARMUP:]
          for p in range(p0, min(p0 + 64, len(cands))):
            v, te = [], []
            for q in range(pairs):
                xa, xb = add_taps(S[p - p0, 2 * q], TAPS), add_taps(S[p - p0, 2 * q + 1], TAPS)
                a, b, _ = held_out(xa, y, x_test=xb, n_test=ce.N_TEST)
                v.append(a)
                te.append(b)
            vals[p, k], tests[p, k] = np.mean(v), np.mean(te)
    g = lambda m: np.exp(np.log(np.maximum(m, 1e-30)).mean(1))  # noqa: E731
    return g(vals), g(tests)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=62)
    ap.add_argument("--noise", type=float, default=1e-3)
    ap.add_argument("--population", type=int, default=512)
    ap.add_argument("--children", type=int, default=240)
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--elite", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--decay", type=float, default=0.75)
    ap.add_argument("--final", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = args.out or f"results/esn_optimize_n{args.size}_noise{args.noise:.0e}"
    t0 = time.time()

    def note(msg):
        print(f"[{time.time() - t0:6.0f} s] {msg}", flush=True)

    # hyper-parameters as in the paper's baseline (grid on the random input weights, series 0)
    best = None
    for r in (0.5, 0.8, 0.95, 1.1):
        for sc in (0.1, 0.3, 1.0):
            w, wi, b = reservoir(args.size, r)
            v, _ = score(w, [(sc * wi, b)], [0], 1, args.noise, salt=1)
            if best is None or v[0] < best[0]:
                best = (v[0], r, sc)
    _, radius, scale = best
    w, wi0, b0 = reservoir(args.size, radius)
    original = (scale * wi0, b0)
    note(f"ESN {args.size} nodes, input noise {args.noise:.0e}: radius {radius}, input scale {scale}")
    gen = torch.Generator().manual_seed(4242)
    rms_w, rms_b = original[0].pow(2).mean().sqrt(), original[1].pow(2).mean().sqrt()
    pop = [original] + [(scale * (2 * torch.rand(args.size, generator=gen, dtype=torch.float64) - 1),
                         0.2 * (2 * torch.rand(args.size, generator=gen, dtype=torch.float64) - 1))
                        for _ in range(args.population - 1)]
    vals, _ = score(w, pop, [0], 1, args.noise, salt=2)
    order = np.argsort(vals)
    elite = [pop[i] for i in order[: args.elite]]
    note(f"generation 0: best {vals[order[0]]:.2e}, elite worst {vals[order[args.elite - 1]]:.2e}")
    sigma = args.sigma
    for g in range(1, args.generations + 1):
        kids = []
        for i in range(args.children):
            a, b = elite[i % len(elite)]
            kids.append((a + sigma * rms_w * torch.randn(args.size, generator=gen, dtype=torch.float64),
                         b + sigma * rms_b * torch.randn(args.size, generator=gen, dtype=torch.float64)))
        cand = elite + kids
        vals, _ = score(w, cand, [0], 1, args.noise, salt=2 + g)
        order = np.argsort(vals)
        elite = [cand[i] for i in order[: args.elite]]
        note(f"generation {g} (sigma {sigma:.3f}): best {vals[order[0]]:.2e}, elite worst {vals[order[args.elite - 1]]:.2e}")
        sigma *= args.decay
    finals = elite[: args.final]
    vals, _ = score(w, finals, [0, 1, 2], 3, args.noise, salt=20)
    chosen = finals[int(np.argmin(vals))]
    note(f"choice on series 0-2: {vals.min():.2e} (validation)")
    rows = []
    for label, c in (("optimised", chosen), ("original", original)):
        _, t = score(w, [c], [3, 4, 5], 3, args.noise, salt=30)
        rows.append({"weights": label, "size": args.size, "input_noise": args.noise, "test_heldout_3_5": t[0]})
        note(f"{label:9s} input weights on held-out series 3-5: {t[0]:.2e}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + ".csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(out + "_weights.json", "w") as f:
        json.dump({"radius": radius, "scale": scale, "w_in": chosen[0].tolist(), "bias": chosen[1].tolist()}, f)
    note("saved " + out + ".csv")


if __name__ == "__main__":
    main()
