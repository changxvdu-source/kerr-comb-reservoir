"""How much noise can a ring take? Scores saved rings under laser, cavity and detector noise.

Each ring is re-simulated under one noise source at a time (and one combined
"realistic" setting), then read out with several detector-noise levels. The
figure of merit is ``nmse_cross``: the readout is trained on one run and tested
on a second run of the same input with independent noise.

Noise levels (see combrc/lle.py for definitions):
  rin        relative rms intensity noise per symbol. For a 1 GHz bandwidth,
             1e-4 / 1e-3 / 1e-2 correspond to about -170 / -150 / -130 dBc/Hz.
  detuning   rms pump-resonance jitter in units of the cavity half-linewidth,
             white ("detuning") or slowly drifting over ~500 symbols ("drift").
  field      intracavity field noise, in multiples of the vacuum-fluctuation level.
  detector   rms noise per comb-line reading, in watts. A photoreceiver with
             10 pW/sqrt(Hz) at 1 GHz bandwidth gives about 3e-7 W.

Baselines see the clean input unless ``input_noise`` is set; for comparison with
laser intensity noise they are also run with input noise rin / modulation depth.

Example:
    python experiments/noise_robustness.py --rings results/train_default_start_256.pt results/untrained_seed0.pt
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from combrc import load_ring, make_series, score_baselines  # noqa: E402
from combrc.evaluate import N_SYMBOLS, WARMUP, score_spectra, simulate, targets  # noqa: E402

DETECTOR_LEVELS = (0.0, 1e-8, 1e-7, 3e-7, 1e-6)
REALISTIC = {"rin": 1e-3, "detuning": 0.01, "field": 1.0}
REALISTIC_DETECTOR = 3e-7
CONDITIONS = (
    [("clean", {})]
    + [("rin", {"rin": v}) for v in (1e-4, 1e-3, 1e-2)]
    + [("detuning", {"detuning": v}) for v in (1e-3, 1e-2, 1e-1)]
    + [("drift", {"detuning": v, "detuning_corr": 500.0}) for v in (1e-2, 1e-1)]
    + [("field", {"field": v}) for v in (1.0, 100.0)]
    + [("realistic", REALISTIC)]
)


def level_of(source, noise):
    if source in ("clean", "realistic"):
        return 0.0
    return noise["detuning" if source == "drift" else source]


def run(job):
    path, source, noise, args = job
    torch.set_num_threads(1)
    u = torch.tensor(make_series(args.task, WARMUP + N_SYMBOLS + args.horizon), dtype=torch.float64)
    ring = load_ring(path)
    settings = dict(noise)
    if "field" in settings:                                   # given in units of the quantum level
        settings["field"] *= ring.quantum_field_noise()
    spectra = simulate(ring.set_noise(**settings), u)
    y = targets(u, args.horizon)
    rows = []
    detectors = DETECTOR_LEVELS if source in ("clean", "realistic") else (0.0,)
    for det in detectors:
        scores = score_spectra(spectra, y, args.n_lines, detector_noise=det)
        rows.append({"ring": os.path.splitext(os.path.basename(path))[0], "source": source,
                     "level": level_of(source, noise), "detector_noise": det, **scores})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rings", nargs="+", required=True, help=".pt files written by save_ring")
    ap.add_argument("--task", default="mackey_glass", choices=["mackey_glass", "rossler", "lorenz"])
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--n-lines", type=int, default=31)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="noise_robustness")
    args = ap.parse_args()

    u = torch.tensor(make_series(args.task, WARMUP + N_SYMBOLS + args.horizon), dtype=torch.float64)
    base = {}
    for input_noise in (0.0, 3e-4, 3e-3, 3e-2):                # = rin / 0.33 modulation depth
        b = score_baselines(u, args.horizon, esn_sizes=(51,), input_noise=input_noise)
        base[input_noise] = {"best_linear": min(v for k, v in b.items() if k.startswith("linear")),
                             "esn51": b["esn51"]}
        print(f"baselines, input noise {input_noise:.0e}: "
              f"best linear {base[input_noise]['best_linear']:.2e}, ESN-51 {base[input_noise]['esn51']:.2e}")

    jobs = [(path, source, noise, args) for path in args.rings for source, noise in CONDITIONS]
    with Pool(args.workers) as pool:
        rows = [row for part in pool.map(run, jobs) for row in part]

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, args.name)
    with open(stem + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rings = list(dict.fromkeys(r["ring"] for r in rows))
    print(f"\n{'condition':34s}" + "".join(f"{name[-22:]:>24s}" for name in rings) + "   (nmse_cross / consistency)")
    seen = []
    for r in rows:
        key = (r["source"], r["level"], r["detector_noise"])
        if key in seen:
            continue
        seen.append(key)
        label = f"{r['source']} {r['level']:g}" if r["level"] else r["source"]
        if r["detector_noise"]:
            label += f" + detector {r['detector_noise']:g} W"
        cells = []
        for name in rings:
            m = next(x for x in rows if x["ring"] == name and (x["source"], x["level"], x["detector_noise"]) == key)
            cells.append(f"{m['nmse_cross']:.2e} / {m['consistency']:.3f}")
        print(f"{label:34s}" + "".join(f"{c:>24s}" for c in cells))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("rin", "laser intensity noise (relative rms)"),
              ("detuning", "detuning jitter (half-linewidths, rms)"),
              ("detector", "detector noise per line (W rms)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (source, xlabel) in zip(axes, panels):
        for i, name in enumerate(rings):
            mine = [r for r in rows if r["ring"] == name]
            clean = next(r for r in mine if r["source"] == "clean" and not r["detector_noise"])
            if source == "detector":
                pts = [(r["detector_noise"], r["nmse_cross"]) for r in mine
                       if r["source"] == "clean" and r["detector_noise"]]
            else:
                pts = [(r["level"], r["nmse_cross"]) for r in mine if r["source"] == source]
            xs, ys = zip(*sorted(pts))
            ax.loglog(xs, ys, "o-", color=f"C{i}", label=name)
            ax.axhline(clean["nmse_cross"], color=f"C{i}", linestyle=":", lw=0.8)
        ax.axhline(base[0.0]["best_linear"], color="gray", linestyle="--", label="best linear AR (clean input)")
        ax.axhline(base[0.0]["esn51"], color="black", linestyle="--", lw=0.8, label="ESN-51 (clean input)")
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(f"cross-run test NMSE ({args.horizon} steps ahead)")
    axes[0].legend(fontsize=7)
    fig.suptitle("dotted: same ring without noise", fontsize=9)
    fig.tight_layout()
    fig.savefig(stem + ".png", dpi=130)
    print("\nsaved", stem + ".csv", "and .png")


if __name__ == "__main__":
    main()
