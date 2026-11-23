"""Classical controllers for the ring-controls-ring tasks, the fair reference for the learned ones.

  PD        c = -k1 e[n] - k2 (e[n] - e[n-1]),  e = observed power - goal (a linear analog loop)
  PD + FF   the same plus the steady-state pump for the goal power, F^2 = Y (1 + (zeta - Y)^2), minus F0.
            The feedforward depends only on the commanded trajectory, which is known in advance, so it can
            be computed beforehand and played out; it needs the plant's model (zeta, F0).
Gains are chosen on separate design trials (loop seed 900, 8 trials) from a fixed grid, then the controllers
run on the same test loops as ring_controls_ring.py (loop seeds 100 + training seed, 16 trials, 1000 steps).
Writes results/ring_controls_ring_classical.csv.

    python experiments/ring_controls_ring_classical.py
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ring_controls_ring as r  # noqa: E402

K1 = (-1.0, -0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2)
K2 = (-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0)


class PD:
    def __init__(self, k1, k2, ff, c_max):
        self.k1, self.k2, self.ff, self.c_max, self.prev = k1, k2, ff, c_max, None

    def __call__(self, y, c_prev, goal):
        e = y - goal
        d = 0 * e if self.prev is None else e - self.prev
        self.prev = e
        c = -self.k1 * e - self.k2 * d
        if self.ff:
            c = c + np.sqrt(goal * (1 + (r.ZETA - goal) ** 2)) - r.F0
        return np.clip(c, -self.c_max, self.c_max)


def score(task, misses, goal):
    return np.sqrt((misses[:, len(goal) // 4:] ** 2).mean()) if task == "hold" else (misses ** 2).mean() / np.var(goal)


def main():
    rows = []
    for task in ("hold", "track"):
        goal = np.full(1000, r.Y_GOAL_HOLD) if task == "hold" else r.commanded(1000)
        c_max = r.C_MAX[task]
        for ff in ((False,) if task == "hold" else (False, True)):
            best = None
            for k1 in K1:
                for k2 in K2:
                    m, _, _ = r.run_loop(lambda plant, k1=k1, k2=k2: PD(k1, k2, ff, c_max), task, goal, 8, seed=900)
                    s = score(task, m, goal)
                    if best is None or s < best[0]:
                        best = (s, k1, k2)
            _, k1, k2 = best
            name = "PD + feedforward" if ff else "PD"
            for seed in range(4):
                m, c, cm = r.run_loop(lambda plant: PD(k1, k2, ff, c_max), task, goal, 16, seed=100 + seed)
                rows.append({"task": task, "controller": name, "k1": k1, "k2": k2, "seed": seed,
                             **r.metrics(task, m, c, goal, cm)})
                print(task, name, k1, k2, seed, {k: round(v, 4) for k, v in rows[-1].items() if isinstance(v, float)})
    fields = ["task", "controller", "k1", "k2", "seed", "within_tolerance", "rms_miss", "mean_control",
              "rms_miss_last_3_quarters", "tracking_nmse"]
    with open(os.path.join(os.path.dirname(HERE), "results", "ring_controls_ring_classical.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
