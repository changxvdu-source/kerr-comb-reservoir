"""Which noise source sets the floor at the operating points chosen in floor_mg_ideal?

The noise-floor comparison (compare_encodings.py, groups comb_mask, mask_sin, mask_bright, mask_drop,
mask_slow2, ideal detectors) found the floor nearly the same for a silica ring, a Si3N4 ring with about
40 times fewer photons per unit of nonlinearity, and the silica ring at 8 times the power. A floor set by
quantum noise would scale with the photon number. Here each chosen point is re-run with one noise source
at a time, on new runs (the fresh-run protocol: readout fitted on one run, scored on the other):

  none, rin, detuning, field   simulation with that optical source only, ideal detectors
  shot                         noiseless simulation, photon shot noise at the detectors
  optical                      rin + detuning + field, ideal detectors
  optical + shot               the floor of the comparison

and also reports the photon numbers that set the quantum noise: mean intracavity power and detected
power per comb line. Same operating point (chosen on series 0) on every series.

    python experiments/floor_breakdown.py --device cuda
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import compare_encodings as ce  # noqa: E402
from combrc import make_task  # noqa: E402
from combrc.evaluate import N_SYMBOLS, WARMUP, score_spectra  # noqa: E402
from combrc.readout import select_lines  # noqa: E402

# condition -> (optical noise tag of the simulation, shot noise at the detectors)
CONDITIONS = {"none": ("none", False), "rin": ("rin", False), "detuning": ("detuning", False),
              "field": ("field", False), "shot": ("none", True), "optical": ("realistic", False),
              "optical + shot": ("realistic", True)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default="results/floor_mg_ideal_summary.csv")
    ap.add_argument("--series-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--variant", default="31x8")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/floor_breakdown")
    args = ap.parse_args()
    variant = tuple(int(x) for x in args.variant.split("x"))
    with open(args.summary) as f:
        chosen = [(r["task"], ce.sample(r["group"], int(r["trial"]))) for r in csv.DictReader(f)]

    rows = []
    for task, cfg in chosen:
        ring = ce.build(cfg, 512, 20)
        photon = 6.626e-34 * 299_792_458 / 1550e-9
        info = {"group": cfg["group"], "trial": cfg["trial"], "p_total_mW": cfg["p_total"] * 1e3,
                "q_total": ring.q_total.item(), "gamma_L": ring.gamma_l}
        errors = {}
        for seed in args.series_seeds:
            _, y = make_task(task, ce.WARMUP + ce.N_TOTAL, seed=seed)
            y = torch.tensor(y, dtype=torch.float64)[WARMUP:]
            sims = {}
            for tag in sorted({t for t, _ in CONDITIONS.values()}):
                sims[tag] = ce.fresh_lines(task, [cfg], 512, 20, seed, tag, args.pairs, args.device, salt=7)[0].double()
            if seed == args.series_seeds[0]:
                lines = sims["none"][0]                       # [symbols, 61] drop-port powers, W
                main = select_lines(lines, variant[0])
                info["detected_uW_per_line"] = main.mean().item() * 1e6
                info["detected_photons_per_line_per_symbol"] = (main.mean().item() * cfg["symbol_time"] / photon)
                # intracavity power from the pump line at the drop port: P_line = theta2 |a_0|^2
                info["drop_fraction"] = (ring.theta2 / (2 * ring.alpha)).item()
            for name, (tag, shot) in CONDITIONS.items():
                det = {"shot_time": cfg["symbol_time"]} if shot else {}
                for k in range(args.pairs):
                    s = score_spectra(sims[tag][2 * k: 2 * k + 2], y, variant[0], taps=variant[1], seed=1000 + k,
                                      n_test=ce.N_TEST, **det)
                    errors.setdefault(name, {}).setdefault(seed, []).append(s["nmse_cross"])
        for name in CONDITIONS:
            info[name] = ce.gmean([np.mean(errors[name][s]) for s in args.series_seeds])
        rows.append(info)
        print(cfg["group"], {k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in info.items()}, flush=True)

    with open(args.out + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\ncross-run test NMSE ({args.variant}), geometric mean over {len(args.series_seeds)} series, "
          f"{args.pairs} new run pairs each")
    print(f"{'group':12s}" + "".join(f"{c:>15s}" for c in CONDITIONS) + f"{'uW/line':>10s}{'photons/line':>14s}")
    for r in rows:
        print(f"{r['group']:12s}" + "".join(f"{r[c]:15.2e}" for c in CONDITIONS)
              + f"{r['detected_uW_per_line']:10.2f}{r['detected_photons_per_line_per_symbol']:14.2e}")
    print("saved", args.out + ".csv")


if __name__ == "__main__":
    main()
