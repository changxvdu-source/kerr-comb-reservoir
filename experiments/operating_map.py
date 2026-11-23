"""Where does a per-line mask work? A map over coupling Q and optical input power.

One mask (that of a draw of compare_encodings.py, ``--base``) is kept fixed, with its detuning
in units of the half-linewidth, modulation depth and split of the power between pump and comb
lines. The input coupling Q and the total input power are scanned on a grid. Every point is
simulated twice with independent seed noise, without noise and with the realistic noise of the
encoding comparison, and scored like there (31 lines; consistency without taps, error with 8).

The figure shows, for both noise settings, the consistency between the two runs, the test NMSE
and the number of comb lines within 60 dB of the strongest: no comb at low power, a coherent
comb in between, chaos (consistency near 0) at high power and Q.

Example:
    python experiments/operating_map.py --base 229 --device cuda
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import compare_encodings as ce  # noqa: E402

VARIANTS = ((31, 1), (31, 8))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="mackey_glass_h10")
    ap.add_argument("--base", type=int, default=229, help="per-line-mask draw whose mask and settings are kept")
    ap.add_argument("--log-q-ext1", type=float, nargs=2, default=[5.6, 7.6], help="range of the input coupling Q")
    ap.add_argument("--power", type=float, nargs=2, default=[3e-3, 1.0], help="range of the total input power, W")
    ap.add_argument("--points", type=int, default=17, help="grid points per axis")
    ap.add_argument("--n-modes", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="cpu", help="'cuda' simulates the whole grid on the GPU in one batch")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="operating_map")
    args = ap.parse_args()
    stem = os.path.join(args.out, args.name)
    cache = stem + "_spectra"
    os.makedirs(cache, exist_ok=True)

    base = ce.sample("comb_mask", args.base)
    qs = np.linspace(*args.log_q_ext1, args.points)
    powers = np.logspace(*np.log10(args.power), args.points)
    cfgs = []
    for i, lq1 in enumerate(qs):
        alpha = ce.cavity(lq1)[0]
        for j, power in enumerate(powers):
            cfgs.append({**base, "trial": i * args.points + j, "mask_seed": args.base, "log_q_ext1": float(lq1),
                         "p_total": float(power), "delta0": base["zeta"] * alpha})
    jobs = [(args.task, cfg, args.n_modes, args.steps, cache, VARIANTS, 0, noise)
            for noise in ("none", "realistic") for cfg in cfgs]
    if args.device != "cpu":
        ce.prefill(jobs, args.device)
    with Pool(args.workers) as pool:
        rows = pool.map(ce.run, jobs, chunksize=4)
    for row, job in zip(rows, jobs):
        row["noise"] = job[-1]
    with open(stem + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = args.points
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for r, noise in enumerate(("none", "realistic")):
        mine = [x for x in rows if x["noise"] == noise]
        q_total = np.array([x["q_total"] for x in mine]).reshape(n, n)[:, 0]
        panels = (("consistency of two runs", np.array([x["consistency_31x1"] for x in mine]), "viridis", (0, 1)),
                  ("log10 test NMSE (31 lines x 8 taps)",
                   np.log10(np.clip([x["test_31x8"] for x in mine], 1e-7, 10.0)), "magma_r", (-6, 0)),
                  ("log10 number of comb lines", np.log10(np.clip([x["comb_lines"] for x in mine], 1, None)),
                   "cividis", (0, np.log10(args.n_modes))))
        for ax, (title, values, cmap, limits) in zip(axes[r], panels):
            mesh = ax.pcolormesh(powers * 1e3, q_total, values.reshape(n, n), cmap=cmap, vmin=limits[0],
                                 vmax=limits[1], shading="nearest")
            ax.plot(base["p_total"] * 1e3, 1 / (1 / 10 ** base["log_q_ext1"] + 1 / 1e8 + 1 / 2.2222e7), "wx")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{title}\nnoise: {noise}", fontsize=9)
            fig.colorbar(mesh, ax=ax)
        axes[r][0].set_ylabel("loaded Q")
    for ax in axes[1]:
        ax.set_xlabel("total optical input power (mW)")
    fig.suptitle(f"{args.task}: mask of draw {args.base}, detuning {base['zeta']:.2f} half-linewidths "
                 f"(cross: the draw itself)", fontsize=10)
    fig.tight_layout()
    fig.savefig(stem + ".png", dpi=130)
    best = min((x for x in rows if x["noise"] == "realistic"), key=lambda x: x["val_31x8"])
    print(f"best point with noise: Q {best['q_total']:.1e}, {best['p_total'] * 1e3:.0f} mW, "
          f"test NMSE {best['test_31x8']:.2e}, consistency {best['consistency_31x1']:.3f}")
    print("saved", stem + ".csv and .png; spectra cached in", cache)


if __name__ == "__main__":
    main()
