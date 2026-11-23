"""How the software baselines depend on the input noise they are given, against the ring's own results.

The paper gives the ESN an input noise of 3e-3 rms: the ring's laser intensity noise (1e-3) referred to the
input through the modulation depth (0.33). This collects the ESN 31 / 100 / 300 results of
strong_baselines.py (same protocol as the paper's baselines: full hyperparameter grid, 3 random reservoirs,
3 noise realisations for the choice, 6 for the test, 3 series) at input noise 0, 1e-4, 3e-4, 1e-3 and 3e-3
into results/esn_input_noise_sweep.csv, prints them next to the ring's results under realistic noise, and
gives the input noise at which each ESN would match the ring (log-log interpolation; '<1e-4' or '>3e-3'
when outside the sweep).

    python experiments/noise_sweep_summary.py
"""
import csv
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
TASKS = ["mackey_glass_h10", "channel_eq", "narma10", "fiber_pam4_17km"]
LEVELS = ["0", "1e-4", "3e-4", "1e-3", "3e-3"]
SIZES = ["ESN 31", "ESN 100", "ESN 300"]
# the ring under realistic noise, fresh-run protocol: (variant, test NMSE, result file)
RING = {"mackey_glass_h10": [("phase mask", "robust_noisy_mg_summary.csv", "comb_mask"),
                             ("stronger drop coupling", "floor_mg_pin_summary.csv", "mask_drop")],
        "channel_eq": [("phase mask", "colab_robust_noisy_ce_long_summary.csv", "comb_mask")],
        "narma10": [("phase mask", "robust_noisy_narma_summary.csv", "comb_mask")],
        "fiber_pam4_17km": [("phase mask", "colab_fiber17_summary.csv", "comb_mask"),
                            ("2 ns symbols", "colab_fiber17_summary.csv", "mask_slow2")]}


def gmean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def read(name):
    path = os.path.join(RES, name)
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def esn_values():
    """{(task, level, model): (gmean NMSE over series, mean SER or nan, number of series)}"""
    per = {}
    for task in TASKS:
        for level in LEVELS[:-1]:
            for s in (0, 1, 2):
                files = [f"colab_sw_{task}_{level}_s{s}.csv"]
                if task == "mackey_glass_h10":     # the later series were run one ESN size per job
                    files += [f"colab_swmg_{level}_s{s}_n{n}.csv" for n in (31, 100, 300)]
                for r in (r for name in files for r in read(name)):
                    if r["model"] in SIZES and (r["noise"] == "none") == (level == "0"):
                        per.setdefault((task, level, r["model"]), []).append((float(r["test_nmse"]), float(r["ser"])))
        for r in read(f"colab_strong_baselines_{task}.csv"):     # 3e-3, and 0 for Mackey-Glass, all 3 series
            if r["model"] in SIZES:
                level = "3e-3" if r["noise"] == "realistic" else "0"
                key = (task, level, r["model"])
                if key not in per:
                    per[key] = [(float(r["test_nmse"]), float(r["ser"]))] * 3
    return {k: (gmean([v[0] for v in vals]), sum(v[1] for v in vals) / len(vals), len(vals)) for k, vals in per.items()}


def crossing(points, target):
    """Input noise at which a curve [(noise > 0, nmse)] reaches ``target`` (log-log interpolation)."""
    pts = sorted(points)
    if not pts:
        return "?"
    if target <= pts[0][1]:
        return f"<{pts[0][0]:.0e}"
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 <= target <= y1:
            f = (math.log(target) - math.log(y0)) / (math.log(y1) - math.log(y0))
            return f"{math.exp(math.log(x0) + f * (math.log(x1) - math.log(x0))):.1e}"
    return f">{pts[-1][0]:.0e}"


def main():
    esn = esn_values()
    rows = [{"task": t, "input_noise": lv, "model": m, "test_nmse": v[0], "ser": v[1], "series": v[2]}
            for (t, lv, m), v in sorted(esn.items(), key=lambda kv: (TASKS.index(kv[0][0]), LEVELS.index(kv[0][1]),
                                                                       SIZES.index(kv[0][2])))]
    with open(os.path.join(RES, "esn_input_noise_sweep.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for task in TASKS:
        print(f"\n{task}: ESN test NMSE vs input noise (series)")
        print("  " + " " * 8 + "".join(f"{lv:>12s}" for lv in LEVELS))
        for m in SIZES:
            cells = [esn.get((task, lv, m)) for lv in LEVELS]
            print(f"  {m:8s}" + "".join(f"{c[0]:9.2e} ({c[2]})" if c else f"{'-':>12s}" for c in cells))
        for label, name, group in RING[task]:
            ring = [float(r["fresh_test"]) for r in read(name) if r["group"] == group]
            if not ring:
                continue
            curve = {m: [(float(lv), esn[(task, lv, m)][0]) for lv in LEVELS[1:] if (task, lv, m) in esn] for m in SIZES}
            print(f"  ring, {label}: {ring[0]:.2e};  ESN matches it at input noise "
                  + ", ".join(f"{m.split()[1]}: {crossing(curve[m], ring[0])}" for m in SIZES))


if __name__ == "__main__":
    main()
