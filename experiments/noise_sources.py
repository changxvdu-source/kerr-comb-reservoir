"""Which noise source costs the comb encodings their accuracy?

compare_encodings.py --noise realistic switches five noise sources on together. Here the
same equal-budget random search is repeated with one source at a time, and with
combinations that correspond to different ways of detecting the comb lines:

  none             no noise
  rin              laser intensity noise, 0.1 % rms per symbol
  detuning         detuning jitter, 1 % of the half-linewidth
  field            quantum-level field noise in the cavity
  shot             photon shot noise at the detectors, 1 ns integration per reading
  det <x> uW       electronic detector noise of <x> uW rms per comb line
  optical          rin + detuning + field
  optical + shot   the above with ideal detectors: the limit of direct detection
  full, det <x>    optical + shot + detector noise ("full, det 0.3" is what
                   compare_encodings.py calls realistic; amplified InGaAs PIN receivers
                   of about 1 GHz bandwidth span 0.13-0.63 uW)
  preamp <g> dB    "full, det 0.3" with an optical amplifier of gain <g> dB and noise
                   figure 5 dB in front of the detectors

Per condition, series and encoding the operating point is chosen on the validation part
and reported on the test part (cross-run NMSE, geometric mean over the series). The
"fixed point" column keeps the operating point chosen without noise instead, which shows
how much one given device suffers rather than how well one can re-optimise around it.

Spectra are shared with compare_encodings.py through its cache; only the single-source
runs (rin, detuning, field) need simulations of their own. Detection noise is added when
the cached spectra are scored.

Example:
    python experiments/noise_sources.py --series-seeds 0 1 2 --workers 16
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import compare_encodings as ce  # noqa: E402
from combrc import make_task  # noqa: E402
from combrc.evaluate import N_SYMBOLS, WARMUP, score_spectra  # noqa: E402
from combrc.tasks import symbol_error_rate  # noqa: E402

SHOT = {"shot_time": ce.REALISTIC["shot_time"]}
PIN = ce.REALISTIC["detector"]                             # W rms per comb line


def det_name(level):
    return f"{level * 1e6:g}"


# name -> (noise tag of the simulation, detector noise, further arguments of readout.measure)
CONDITIONS = {"none": ("none", 0.0, {}), "rin": ("rin", 0.0, {}), "detuning": ("detuning", 0.0, {}),
              "field": ("field", 0.0, {}), "shot": ("none", 0.0, SHOT)}
CONDITIONS.update({f"det {det_name(d)} uW": ("none", d, {}) for d in (3e-8, 1e-7, 3e-7)})
CONDITIONS.update({"optical": ("realistic", 0.0, {}), "optical + shot": ("realistic", 0.0, SHOT)})
CONDITIONS.update({f"full, det {det_name(d)}": ("realistic", d, SHOT) for d in (3e-8, 1e-7, 3e-7, 6e-7)})
CONDITIONS.update({f"preamp {g} dB": ("realistic", PIN, {**SHOT, "preamp_gain_db": g, "preamp_nf_db": 5})
                   for g in (10, 20)})


def run(job):
    task, cfg, n_modes, steps, cache, variants, seed, optical = job
    torch.set_num_threads(1)
    lines, q_total, comb_lines = ce.spectra_of(task, cfg, n_modes, steps, cache, seed, optical)
    _, y = make_task(task, WARMUP + N_SYMBOLS, seed=seed)
    y = torch.tensor(y, dtype=torch.float64)[WARMUP:]
    rows = []
    for name, (tag, detector, detection) in CONDITIONS.items():
        if tag != optical:
            continue
        row = {"task": task, "series_seed": seed, "condition": name, "group": cfg["group"], "trial": cfg["trial"],
               "q_total": q_total, "p_total": cfg["p_total"], "comb_lines": comb_lines}
        for variant in variants:
            try:
                if comb_lines < 0:
                    raise FloatingPointError
                s = score_spectra(lines, y, variant[0], taps=variant[1], detector_noise=detector,
                                  error_rate=symbol_error_rate if task == "channel_eq" else None, **detection)
                scores = (s["val_nmse_cross"], s["nmse_cross"], s["consistency"])
            except (FloatingPointError, RuntimeError):      # diverged simulation or singular fit
                scores = (float("inf"), float("inf"), 0.0)
            row.update(dict(zip((f"{p}_{ce.tag(variant)}" for p in ("val", "test", "consistency")), scores)))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=32)
    ap.add_argument("--tasks", nargs="+", default=["mackey_glass_h10", "channel_eq"])
    ap.add_argument("--groups", nargs="+", default=["comb_mask", "comb_uniform"], choices=ce.GROUPS)
    ap.add_argument("--series-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=["31x1", "31x8"])
    ap.add_argument("--n-modes", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="noise_sources")
    ap.add_argument("--cache", default=os.path.join("results", "encodings_spectra"))
    args = ap.parse_args()
    variants = tuple(tuple(int(x) for x in v.split("x")) for v in args.variants)
    seeds = args.series_seeds
    os.makedirs(args.cache, exist_ok=True)

    jobs = [(task, ce.sample(group, trial), args.n_modes, args.steps, args.cache, variants, seed, optical)
            for optical in sorted({c[0] for c in CONDITIONS.values()})
            for seed in seeds for task in args.tasks for group in args.groups for trial in range(args.trials)]
    with Pool(args.workers) as pool:
        rows = [r for chunk in pool.map(run, jobs, chunksize=1) for r in chunk]
    stem = os.path.join(args.out, args.name)
    with open(stem + "_trials.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    index = {}
    for r in rows:
        index.setdefault((r["task"], r["condition"], r["group"], r["series_seed"]), []).append(r)

    summary = []
    for task in args.tasks:
        base = {noise: ce.baselines(task, variants, seeds, noise) for noise in ("none", "realistic")}
        for variant in variants:
            vt = ce.tag(variant)
            print(f"\n=== {task}, readout {vt}: cross-run test NMSE, geometric mean over {len(seeds)} series")
            print(f"    ESN of the same size: {base['none']['esn_' + vt]:.2e} without noise, "
                  f"{base['realistic']['esn_' + vt]:.2e} with input noise {ce.REALISTIC['baseline_input_noise']:g}"
                  f";  best linear AR {base['none']['linear_best']:.2e}")
            print(f"{'condition':16s}" + "".join(f"{g + ': re-chosen':>26s}{'min..max':>19s}{'fixed point':>13s}"
                                                 f"{'consist.':>9s}{'  chosen on series ' + str(seeds[0]):28s}"
                                                 for g in args.groups))
            for name in CONDITIONS:
                line = f"{name:16s}"
                for group in args.groups:
                    chosen = [min(index[(task, name, group, s)], key=lambda r: r[f"val_{vt}"]) for s in seeds]
                    clean = [min(index[(task, "none", group, s)], key=lambda r: r[f"val_{vt}"]) for s in seeds]
                    fixed = [next(r for r in index[(task, name, group, s)] if r["trial"] == c["trial"])
                             for s, c in zip(seeds, clean)]
                    tests = [r[f"test_{vt}"] for r in chosen]
                    entry = {"task": task, "readout": vt, "condition": name, "group": group,
                             "test": ce.gmean(tests), "min_test": min(tests), "max_test": max(tests),
                             "test_fixed_point": ce.gmean([r[f"test_{vt}"] for r in fixed]),
                             "consistency": float(np.mean([r[f"consistency_{vt}"] for r in chosen])),
                             "trial": chosen[0]["trial"], "q_total": chosen[0]["q_total"],
                             "p_total": chosen[0]["p_total"],
                             "esn_clean": base["none"]["esn_" + vt], "esn_input_noise": base["realistic"]["esn_" + vt],
                             "linear": base["none"]["linear_best"]}
                    summary.append(entry)
                    span = f"{entry['min_test']:.1e}..{entry['max_test']:.1e}"
                    where = f"  trial {entry['trial']}, Q {entry['q_total']:.1e}, {entry['p_total'] * 1e3:.0f} mW"
                    line += (f"{entry['test']:26.2e}{span:>19s}{entry['test_fixed_point']:13.2e}"
                             f"{entry['consistency']:9.3f}{where:28s}")
                print(line)
    with open(stem + "_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vt = ce.tag(variants[-1])
    fig, axes = plt.subplots(1, len(args.tasks), figsize=(6.5 * len(args.tasks), 4.6), squeeze=False)
    x = np.arange(len(CONDITIONS))
    width = 0.8 / len(args.groups)
    for ax, task in zip(axes[0], args.tasks):
        for i, group in enumerate(args.groups):
            mine = [s for s in summary if (s["task"], s["readout"], s["group"]) == (task, vt, group)]
            best = np.array([s["test"] for s in mine])
            err = np.array([[max(b - s["min_test"], 0), max(s["max_test"] - b, 0)] for b, s in zip(best, mine)]).T
            ax.bar(x - 0.4 + (i + 0.5) * width, best, width, yerr=err, capsize=2, label=group)
        ax.axhline(mine[0]["esn_clean"], color="black", linestyle=":", label="ESN, no noise")
        ax.axhline(mine[0]["esn_input_noise"], color="black", linestyle="--", label="ESN, noisy input")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(list(CONDITIONS), rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{task}, readout {vt}")
    axes[0][0].set_ylabel(f"cross-run test NMSE, best of {args.trials}, mean of {len(seeds)} series")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(stem + ".png", dpi=130)
    print("\nsaved", stem + "_trials.csv, _summary.csv and .png")


if __name__ == "__main__":
    main()
