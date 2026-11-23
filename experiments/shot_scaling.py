"""Controlled check: how does the shot-noise part of the error fall with the number of detected photons?

Same operating point, same simulated runs (the best single ring of readout_quantization.py: stronger drop
coupling, trial 211, realistic optical noise, line powers saved in results/readout_quantization_lines_s*.npy).
Ideal detectors (no electronic noise); only the photon count per reading is scaled, by multiplying the shot-noise
integration time by ``k`` (equivalent to k times more light on every detector at the same comb state).
The shot-noise contribution is the error with shot noise minus the error without it.

    python experiments/shot_scaling.py
"""
import csv
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import compare_encodings as ce  # noqa: E402
from combrc import make_task, to_features  # noqa: E402
from combrc.evaluate import WARMUP, held_out  # noqa: E402
from combrc.readout import add_taps, detector_floor, measure  # noqa: E402

TAPS, PAIRS = 8, 6
FACTORS = [None, 1, 3, 10, 30, 100]


def main():
    cfg = ce.sample("mask_drop", 211)
    t = cfg.get("symbol_time", 1e-9)
    floor = detector_floor(0.0)
    per = {k: [] for k in FACTORS}
    for seed in (0, 1, 2):
        _, y_np = make_task("mackey_glass_h10", WARMUP + ce.N_TOTAL, seed=seed)
        y = torch.tensor(y_np, dtype=torch.float64)[WARMUP: WARMUP + ce.N_TOTAL]
        lines = torch.tensor(np.load(os.path.join("results", f"readout_quantization_lines_s{seed}.npy")))
        for k in FACTORS:
            tests = []
            for p in range(PAIRS):
                gen = torch.Generator().manual_seed(1000 + p)
                shot = 0.0 if k is None else k * t
                xa, xb = (add_taps(to_features(measure(lines[2 * p + i], 0.0, gen, shot_time=shot), floor=floor), TAPS)
                          for i in (0, 1))
                tests.append(held_out(xa, y, x_test=xb, n_test=ce.N_TEST)[1])
            per[k].append(float(np.mean(tests)))
    base = ce.gmean(per[None])
    rows = []
    for k in FACTORS:
        total = ce.gmean(per[k])
        rows.append({"photons_x": "no shot noise" if k is None else k, "test_nmse": total,
                     "shot_part": "" if k is None else total - base})
        print(f"{rows[-1]['photons_x']!s:>14}  {total:.3e}  shot part {rows[-1]['shot_part']!s}")
    with open(os.path.join("results", "shot_scaling.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
