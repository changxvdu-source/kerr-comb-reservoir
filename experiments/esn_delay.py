"""The echo state network with the same delay line as the ring's ``mask_delay`` group.

The ring with a delay line sees u[t] and u[t - d] together; the fair software reference is an ESN
whose input also carries u[t - d] (a second input weight vector), with d chosen on the validation
part, like the ring's random search chooses it. Same protocol as ``compare_encodings.esn_fresh``:
hyper-parameters (spectral radius, input scale, d) chosen by the mean validation error over
``--select-pairs`` realisations of the input noise, test error over ``--pairs`` new realisations,
geometric mean over the series. Also reported: d fixed to each value, to see which delay helps.

    python experiments/esn_delay.py --tasks narma10 --noise realistic
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from combrc import fit_readout, make_task, nmse  # noqa: E402
from combrc.evaluate import N_SYMBOLS, WARMUP, _splits  # noqa: E402
from combrc.readout import add_taps  # noqa: E402
from compare_encodings import REALISTIC, gmean  # noqa: E402

DELAYS = range(0, 13)          # 0 = no delay line (the ordinary ESN)


def esn2_states(u, u2, n_nodes, spectral_radius, input_scale, seed=0):
    """tanh ESN driven by two inputs ([T] each); the second input gets its own random weights."""
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(n_nodes, n_nodes, generator=gen, dtype=torch.float64)
    w = w * (spectral_radius / torch.linalg.eigvals(w).abs().max())
    w_in = input_scale * (2 * torch.rand(n_nodes, generator=gen, dtype=torch.float64) - 1)
    bias = 0.2 * (2 * torch.rand(n_nodes, generator=gen, dtype=torch.float64) - 1)
    w_in2 = input_scale * (2 * torch.rand(n_nodes, generator=gen, dtype=torch.float64) - 1)
    x = torch.zeros(n_nodes, dtype=torch.float64)
    out = []
    for a, b in zip(2 * u - 1, 2 * u2 - 1):
        x = torch.tanh(w @ x + w_in * a + w_in2 * b + bias)
        out.append(x)
    return torch.stack(out)


def shifted(u, d):
    return u if d == 0 else torch.cat([u[:1].expand(d), u[:-d]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=["narma10"])
    ap.add_argument("--series-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--noise", default="realistic", choices=["none", "realistic"])
    ap.add_argument("--size", type=int, default=31)
    ap.add_argument("--taps", type=int, default=12)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--select-pairs", type=int, default=3)
    args = ap.parse_args()
    input_noise = REALISTIC["baseline_input_noise"] if args.noise == "realistic" else 0.0
    n_train, n_val = _splits(N_SYMBOLS)
    val_part, test_part = slice(n_train, n_train + n_val), slice(n_train + n_val, None)

    for task in args.tasks:
        chosen, fixed = [], {d: [] for d in DELAYS}
        for seed in args.series_seeds:
            u, y = (torch.tensor(a, dtype=torch.float64) for a in make_task(task, WARMUP + N_SYMBOLS, seed=seed))
            target = y[WARMUP: WARMUP + N_SYMBOLS]

            def errors(d, radius, scale, noise_seed):
                gen = torch.Generator().manual_seed(noise_seed)
                seen = u + input_noise * torch.randn(u.shape, generator=gen, dtype=u.dtype) if input_noise else u
                seen = seen[: WARMUP + N_SYMBOLS]
                states = add_taps(esn2_states(seen, shifted(seen, d), args.size, radius, scale), args.taps)[WARMUP:]
                pred = fit_readout(states, target, n_train, n_val).predict(states)
                return nmse(target[val_part], pred[val_part]).item(), nmse(target[test_part], pred[test_part]).item()

            grid = [(r, s) for r in (0.5, 0.8, 0.95, 1.1) for s in (0.1, 0.3, 1.0)]
            val = {(d, r, s): np.mean([errors(d, r, s, k)[0] for k in range(args.select_pairs)])
                   for d in DELAYS for r, s in grid}
            for d in DELAYS:          # best hyper-parameters for each fixed delay
                _, r, s = min((val[(d, r, s)], r, s) for r, s in grid)
                fixed[d].append(np.mean([errors(d, r, s, 100 + k)[1] for k in range(args.pairs)]))
            best = min(val, key=val.get)
            chosen.append((best[0], np.mean([errors(*best, 100 + k)[1] for k in range(args.pairs)])))
            print(f"{task} series {seed}: chosen d={best[0]}, test {chosen[-1][1]:.3e}", flush=True)
        print(f"\n=== {task}, noise {args.noise}, ESN {args.size} nodes, {args.taps} taps, geometric mean over "
              f"{len(args.series_seeds)} series, {args.pairs} new noise realisations each")
        print(f"delay chosen on validation (d = {[c[0] for c in chosen]}): test NMSE {gmean([c[1] for c in chosen]):.3e}")
        for d in DELAYS:
            print(f"  d = {d:2d}: {gmean(fixed[d]):.3e}")


if __name__ == "__main__":
    main()
