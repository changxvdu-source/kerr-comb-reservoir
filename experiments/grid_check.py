"""Is a noisy result grid-converged, or does it just vary from one noise realisation to the next?

Each ring is simulated several times with independent noise on four grids (modes x Runge-Kutta
steps per symbol). The runs are scored in pairs exactly like the encoding comparison (readout
fitted on one run, tested on the other; 31 lines x 8 taps; shot and detector noise). If the
spread between pairs on one grid is as large as the difference between grids, the grids agree.

Rings are per-line-mask draws of compare_encodings.py (``--draws``) and rings saved by
train_physical.py (``--rings``). Meant for a GPU: all runs of all rings go through as one batch.

Example:
    python experiments/grid_check.py --draws 229 18 --rings results/finetune_runs/train_t018_s1.pt --device cuda
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import compare_encodings as ce  # noqa: E402
from combrc import load_ring, make_task, select_lines, stack_rings  # noqa: E402
from combrc.evaluate import N_SYMBOLS, WARMUP, score_spectra  # noqa: E402

GRIDS = ((512, 20), (512, 40), (1024, 20), (1024, 40))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="mackey_glass_h10")
    ap.add_argument("--draws", type=int, nargs="*", default=[229, 18])
    ap.add_argument("--rings", nargs="*", default=[])
    ap.add_argument("--pairs", type=int, default=6, help="independent pairs of runs per ring and grid")
    ap.add_argument("--series-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join("results", "grid_check.csv"))
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    noise = ce.REALISTIC
    u, y = (torch.tensor(a, dtype=torch.float64) for a in make_task(args.task, WARMUP + N_SYMBOLS, seed=args.series_seed))
    names = [f"draw {d}" for d in args.draws] + [os.path.basename(p) for p in args.rings]
    rows = 2 * args.pairs
    lines_out = ["ring,n_modes,steps,pair,test_nmse"]
    print(f"{args.task}, series {args.series_seed}, realistic noise, 31x8: test NMSE over {args.pairs} pairs of runs")
    for n_modes, steps in GRIDS:
        rings = [ce.build(ce.sample("comb_mask", d), n_modes, steps) for d in args.draws]
        rings += [load_ring(p, n_modes=n_modes, steps_per_symbol=steps, trainable=[]) for p in args.rings]
        state = torch.cat([ring.init_state(rows, seed=17 + i) for i, ring in enumerate(rings)]).to(args.device)
        stack = ce.with_noise(stack_rings(rings, repeats=rows).to(args.device), "realistic")
        generator = torch.Generator(device=args.device).manual_seed(1000 * n_modes + steps)
        spectra, _ = stack(u.to(args.device).expand(len(rings) * rows, -1), state=state, generator=generator)
        lines = select_lines(spectra[:, WARMUP:], ce.CACHED_LINES).float().double().cpu()     # rounded like the cache
        del spectra
        for i, name in enumerate(names):
            mine = lines[i * rows: (i + 1) * rows]
            tests = [score_spectra(mine[2 * k: 2 * k + 2], y[WARMUP:], 31, taps=8, detector_noise=noise["detector"],
                                   shot_time=noise["shot_time"], seed=k)["nmse_cross"] for k in range(args.pairs)]
            lines_out += [f"{name},{n_modes},{steps},{k},{t}" for k, t in enumerate(tests)]
            print(f"  {n_modes:4d} modes x {steps} steps   {name:22s} mean {np.mean(tests):.2e}   "
                  f"std {np.std(tests, ddof=1):.1e}   range {min(tests):.2e}..{max(tests):.2e}", flush=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines_out) + "\n")
    print("saved", args.out)


if __name__ == "__main__":
    main()
