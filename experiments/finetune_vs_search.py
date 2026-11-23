"""Does gradient fine-tuning beat random search? Per-line mask, Mackey-Glass 10 steps ahead.

Everything is scored the way compare_encodings.py --noise realistic scores it: 512 modes,
31 comb lines x 8 readout taps, laser, detuning, field, shot and detector noise, cross-run
test NMSE, geometric mean over the series.

  1. Random search: ``--draws`` operating points (the first 32 are those of compare_encodings.py).
  2. Fine-tuning: starting points are taken from the first 32 draws, ranked on the validation
     part of the first series: the ``--top`` best, each with ``--top-seeds`` training seeds, and
     ``--middle`` points from the middle of the ranking with one seed. train_physical.py then
     trains the detuning, input coupling, pump power, modulation depth and the 16 complex mask
     amplitudes under the same noise, on a separate training series, on a 256-mode grid, with
     the optical input power capped at the search's upper limit. It keeps the parameters that
     score best on its own validation series, so a run that never improves returns its start.
  3. Every trained ring is re-simulated on the 512-mode grid on the search's series.

Compared at the end: the best of 32 draws, the best of all draws, and the best of the
fine-tuned rings (each chosen per series on its validation part, like the search does).

Training runs one at a time (each needs about 4 GB) and is skipped when its result file exists.

Example:
    python experiments/finetune_vs_search.py --workers 16
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import compare_encodings as ce  # noqa: E402
from combrc import load_ring, make_task, select_lines  # noqa: E402
from combrc.evaluate import N_SYMBOLS, WARMUP, score_spectra, simulate  # noqa: E402

TASK, GROUP, VARIANT = "mackey_glass_h10", "comb_mask", (31, 8)
VT = ce.tag(VARIANT)
NOISE = ce.REALISTIC
TRAINED = ["delta0", "log_q_ext1", "p_avg", "p_swing", "input_bias", "input_weight"]
MAX_POWER = 0.3                                   # W, the upper end of the search range


def evaluate_ring(job):
    """Score a saved ring on one series, on the search's grid and with the search's noise."""
    path, seed, n_modes, steps = job
    torch.set_num_threads(1)
    ring = load_ring(path, n_modes=n_modes, steps_per_symbol=steps, trainable=[])
    ring.set_noise(rin=NOISE["rin"], detuning=NOISE["detuning"], field=NOISE["field"] * ring.quantum_field_noise())
    u, y = (torch.tensor(a, dtype=torch.float64) for a in make_task(TASK, WARMUP + N_SYMBOLS, seed=seed))
    lines = select_lines(simulate(ring, u), ce.CACHED_LINES).float().double()     # rounded like the search's cache
    s = score_spectra(lines, y[WARMUP:], VARIANT[0], taps=VARIANT[1], detector_noise=NOISE["detector"],
                      shot_time=NOISE["shot_time"])
    d = ring.describe()
    return {"path": path, "series_seed": seed, "val": s["val_nmse_cross"], "test": s["nmse_cross"],
            "consistency": s["consistency"], "q_total": d["q_total"], "power": d["mean_input_power"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=256, help="random operating points in the extended search")
    ap.add_argument("--series-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--top", type=int, default=4, help="best starting points to fine-tune")
    ap.add_argument("--top-seeds", type=int, default=2, help="training seeds per top starting point")
    ap.add_argument("--middle", type=int, default=4, help="starting points from the middle of the ranking")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--lr-scale", type=float, default=0.3)
    ap.add_argument("--train-modes", type=int, default=256)
    ap.add_argument("--n-modes", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--train-threads", type=int, default=4)
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="finetune")
    ap.add_argument("--cache", default=os.path.join("results", "encodings_spectra"))
    args = ap.parse_args()
    seeds = args.series_seeds
    runs_dir = os.path.join(args.out, args.name + "_runs")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)

    # 1. random search
    jobs = [(TASK, ce.sample(GROUP, trial), args.n_modes, args.steps, args.cache, (VARIANT,), seed, "realistic")
            for seed in seeds for trial in range(args.draws)]
    with Pool(args.workers) as pool:
        search = pool.map(ce.run, jobs, chunksize=1)
    row = {(r["series_seed"], r["trial"]): r for r in search}

    def searched(n_draws):
        """Test NMSE per series of the draw with the best validation error among the first n_draws."""
        picks = [min((row[(s, t)] for t in range(n_draws)), key=lambda r: r[f"val_{VT}"]) for s in seeds]
        return [p[f"test_{VT}"] for p in picks], [p["trial"] for p in picks]

    ranking = sorted(range(32), key=lambda t: row[(seeds[0], t)][f"val_{VT}"])
    mid = (32 - args.middle) // 2
    starts = [(t, s, "top") for t in ranking[:args.top] for s in range(args.top_seeds)]
    starts += [(t, 0, "middle") for t in ranking[mid: mid + args.middle]]
    print(f"random search: {args.draws} draws on {len(seeds)} series; validation ranking of the first 32 on "
          f"series {seeds[0]}: {ranking}", flush=True)

    # 2. fine-tuning, one run at a time
    runs = []
    for trial, seed, kind in starts:
        name = f"t{trial:03d}_s{seed}"
        stem = os.path.join(runs_dir, "train_" + name)
        if not os.path.exists(stem + ".json"):
            cmd = [sys.executable, os.path.join(HERE, "train_physical.py"), "--name", name, "--out", runs_dir,
                   "--init-trial", str(trial), "--seed", str(seed), "--train", *TRAINED,
                   "--max-power", str(MAX_POWER), "--rin", str(NOISE["rin"]), "--detuning-noise", str(NOISE["detuning"]),
                   "--field-noise", str(NOISE["field"]), "--detector-noise", str(NOISE["detector"]),
                   "--shot-time", str(NOISE["shot_time"]), "--taps", str(VARIANT[1]), "--n-lines", str(VARIANT[0]),
                   "--lr-scale", str(args.lr_scale), "--iters", str(args.iters), "--eval-every", "10",
                   "--threads", str(args.train_threads), "--n-modes", str(args.train_modes), "--steps", str(args.steps),
                   "--no-fine-grid"]
            with open(stem + ".log", "w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        with open(stem + ".json") as f:
            best_iter = json.load(f)["best_iter"]
        runs.append({"trial": trial, "train_seed": seed, "kind": kind, "rank": ranking.index(trial) + 1,
                     "best_iter": best_iter, "path": stem + ".pt"})
        print(f"trained {name} ({kind}, rank {runs[-1]['rank']}): best validation at iteration {best_iter}", flush=True)

    # 3. the trained rings on the search's grid and series
    with Pool(min(args.workers, 8)) as pool:
        scored = pool.map(evaluate_ring, [(r["path"], s, args.n_modes, args.steps) for r in runs for s in seeds],
                          chunksize=1)
    for r in runs:
        mine = [e for e in scored if e["path"] == r["path"]]
        start = [row[(s, r["trial"])] for s in seeds]
        r.update({"start_test": ce.gmean([x[f"test_{VT}"] for x in start]),
                  "trained_test": ce.gmean([e["test"] for e in mine]),
                  "start_power_mw": start[0]["p_total"] * 1e3, "trained_power_mw": mine[0]["power"] * 1e3,
                  "start_q": start[0]["q_total"], "trained_q": mine[0]["q_total"],
                  "trained_consistency": float(np.mean([e["consistency"] for e in mine]))})
        r["ratio"] = r["trained_test"] / r["start_test"]
        r.update({f"trained_val_s{e['series_seed']}": e["val"] for e in mine})
        r.update({f"trained_test_s{e['series_seed']}": e["test"] for e in mine})

    stem = os.path.join(args.out, args.name)
    with open(stem + "_runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)

    print(f"\n=== {TASK}, {GROUP}, readout {VT}, realistic noise: cross-run test NMSE, geometric mean over "
          f"{len(seeds)} series")
    print(f"{'start':>22s}{'seed':>5s}{'best it':>8s}{'start':>10s}{'trained':>10s}{'ratio':>7s}"
          f"{'power mW':>14s}{'Q':>20s}{'consist.':>9s}")
    for r in runs:
        print(f"{'trial %d (%s, rank %d)' % (r['trial'], r['kind'], r['rank']):>22s}{r['train_seed']:5d}"
              f"{r['best_iter']:8d}{r['start_test']:10.2e}{r['trained_test']:10.2e}{r['ratio']:7.2f}"
              f"{'%.0f -> %.0f' % (r['start_power_mw'], r['trained_power_mw']):>14s}"
              f"{'%.1e -> %.1e' % (r['start_q'], r['trained_q']):>20s}{r['trained_consistency']:9.3f}")
    for kind in ("top", "middle"):
        mine = [r for r in runs if r["kind"] == kind]
        if mine:
            ratios = [r["ratio"] for r in mine]
            print(f"{kind}: {sum(x < 1 for x in ratios)} of {len(ratios)} runs end below their start; "
                  f"geometric mean ratio {ce.gmean(ratios):.2f}, range {min(ratios):.2f}..{max(ratios):.2f}")

    def tuned(kinds):
        """Like ``searched``: per series, the fine-tuned ring with the best validation error."""
        mine = [r for r in runs if r["kind"] in kinds]
        return [min(mine, key=lambda r: r[f"trained_val_s{s}"])[f"trained_test_s{s}"] for s in seeds]

    methods = [("random search, 32 draws", searched(32)[0]),
               (f"random search, {args.draws} draws", searched(args.draws)[0])]
    if args.top:
        methods.append(("32 draws + fine-tuning of the best", tuned(("top",))))
    if args.middle:
        methods.append(("fine-tuning of middle-ranked draws", tuned(("middle",))))
    summary = []
    for label, tests in methods:
        summary.append({"method": label, "test": ce.gmean(tests), "min_test": min(tests), "max_test": max(tests)})
        print(f"{label:38s} {summary[-1]['test']:.2e}   ({min(tests):.1e}..{max(tests):.1e})")
    print(f"draws chosen by the search: {searched(32)[1]} of 32, {searched(args.draws)[1]} of {args.draws}")
    with open(stem + "_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print("\nsaved", stem + "_runs.csv and _summary.csv; training logs in", runs_dir)


if __name__ == "__main__":
    main()
