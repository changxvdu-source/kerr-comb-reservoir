"""Tracking NMSE of the ring-controls-ring runs over training seeds (detection noise 0.01, 3000-step record,
1000 closed-loop steps, 16 trials): seed 0 from the local runs, seeds 1-3 from Colab. Writes
results/ring_controls_ring_seeds.csv with one row per controller (geometric mean and per-seed values).

    python experiments/ring_controls_ring_summary.py
"""
import csv
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FILES = {0: ["rcr_track_software.csv", "rcr_track_esn100_det0.01.csv", "rcr_track_ring.csv"],
         **{s: [f"colab_rcr_track_software_s{s}.csv", f"colab_rcr_track_ring_s{s}.csv"] for s in (1, 2, 3)}}
ORDER = ["exact", "linear", "NG-RC", "ESN 31", "ESN 100", "ring mask_slow2, PIN", "ring mask_slow2, no noise"]


def main():
    per = {}
    for seed, names in FILES.items():
        for name in names:
            for r in csv.DictReader(open(os.path.join(RES, name))):
                if r["controller"] in ORDER:
                    per.setdefault(r["controller"], {})[seed] = float(r["tracking_nmse"])
    rows = []
    for k in ORDER:
        vals = [per[k][s] for s in sorted(per[k])]
        rows.append({"controller": k, "seeds": len(vals),
                     "tracking_nmse_gmean": math.exp(sum(map(math.log, vals)) / len(vals)),
                     "per_seed": " ".join(f"{v:.4f}" for v in vals)})
        print(f"{k:28s} {rows[-1]['tracking_nmse_gmean']:.4f}   ({rows[-1]['per_seed']})")
    with open(os.path.join(RES, "ring_controls_ring_seeds.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
