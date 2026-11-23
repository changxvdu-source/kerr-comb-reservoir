"""The closed loop (Henon map, predict then decide) against the input noise of the software baselines.

The linear predictor, NG-RC and the 31-node ESN of closed_loop_control.py read the plant with input noise
3e-3 of the symbol range by default (the ring's source noise referred to the input). Here they were rerun
alone (``--groups`` with no ring) at input noise 0, 1e-4, 3e-4, 1e-3 and 3e-3, with the training records,
start states and trials of the ring runs (results/cl_sweep/cl_in_<task>_s<seed>_<level>.csv). The rings do
not see this noise, so their results are taken from the earlier runs:

  track, seed 0      closed_loop_track_readout.csv        8 trials
  track, seeds 1-3   colab_cl_henon_track_seed<s>.csv     32 trials each
  hold, seed 0       closed_loop_hold_readout.csv         8 trials
  hold, seed 1       colab_cl_henon_hold_seed1.csv        32 trials

Tracking: geometric mean of the tracking NMSE over the 4 training records. Holding: share of steps within
tolerance and losses of the fixed point per 1000 steps, mean over the 2 records. Writes
results/closed_loop_input_noise.csv.

    python experiments/closed_loop_noise_summary.py
"""
import csv
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
LEVELS = ["0", "1e-4", "3e-4", "1e-3", "3e-3"]
BASELINES = ["linear predictor", "NG-RC", "ESN, 31 nodes"]
RINGS = {"track": {0: "closed_loop_track_readout.csv", 1: "colab_cl_henon_track_seed1.csv",
                   2: "colab_cl_henon_track_seed2.csv", 3: "colab_cl_henon_track_seed3.csv"},
         "hold": {0: "closed_loop_hold_readout.csv", 1: "colab_cl_henon_hold_seed1.csv"}}
RING_ROWS = ["ring comb_mask, PIN detectors", "ring mask_slow2, PIN detectors", "ring mask_slow2, no noise"]
METRICS = {"track": ["tracking_nmse"], "hold": ["within_tolerance", "losses_per_1000_steps", "rms_miss_while_held"]}


def read(path):
    return {r["controller"]: r for r in csv.DictReader(open(path))} if os.path.exists(path) else {}


def combine(task, metric, values):
    values = [v for v in values if v == v]
    if not values:
        return float("nan")
    if task == "track":
        return math.exp(sum(math.log(v) for v in values) / len(values))
    return sum(values) / len(values)


def main():
    rows = []
    for task, ring_files in RINGS.items():
        seeds = sorted(ring_files)
        for model in BASELINES:
            for lv in LEVELS:
                per = [read(os.path.join(RES, "cl_sweep", f"cl_in_{task}_s{s}_{lv}.csv")).get(model) for s in seeds]
                if all(per):
                    rows.append({"task": task, "controller": model, "input_noise": lv, "seeds": len(per),
                                 **{m: combine(task, m, [float(r[m]) for r in per]) for m in METRICS[task]},
                                 "per_seed": " ".join(f"{float(r[METRICS[task][0]]):.3g}" for r in per)})
        for model in RING_ROWS:
            per = [read(os.path.join(RES, ring_files[s])).get(model) for s in seeds]
            if all(per):
                rows.append({"task": task, "controller": model, "input_noise": "-", "seeds": len(per),
                             **{m: combine(task, m, [float(r[m]) for r in per]) for m in METRICS[task]},
                             "per_seed": " ".join(f"{float(r[METRICS[task][0]]):.3g}" for r in per)})
    fields = ["task", "controller", "input_noise", "seeds", "tracking_nmse", "within_tolerance",
              "losses_per_1000_steps", "rms_miss_while_held", "per_seed"]
    with open(os.path.join(RES, "closed_loop_input_noise.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)

    print("tracking NMSE (geometric mean over 4 training records)")
    print(f"  {'':28s}" + "".join(f"{lv:>10s}" for lv in LEVELS))
    for model in BASELINES:
        cells = {r["input_noise"]: r["tracking_nmse"] for r in rows if r["task"] == "track" and r["controller"] == model}
        print(f"  {model:28s}" + "".join(f"{cells[lv]:10.2e}" if lv in cells else f"{'-':>10s}" for lv in LEVELS))
    for r in rows:
        if r["task"] == "track" and r["input_noise"] == "-":
            print(f"  {r['controller']:28s}{r['tracking_nmse']:10.2e}   ({r['per_seed']})")
    print("\nholding the fixed point (mean over 2 training records): within tolerance / losses per 1000 steps / "
          "rms miss while held")
    for r in rows:
        if r["task"] == "hold":
            print(f"  {r['controller']:30s} {r['input_noise']:>5s}   {r['within_tolerance']:.4f}   "
                  f"{r['losses_per_1000_steps']:7.2f}   {r['rms_miss_while_held']:.2e}")


if __name__ == "__main__":
    main()
