"""Scaling of ring arrays against software reservoirs on Mackey-Glass (realistic noise, 16320 training +
validation symbols, 1000 test symbols, 4 pairs of fresh runs per series, 3 series).

Collects the Colab runs into results/scaling_mg.csv, one row per model and size, test NMSE as the
geometric mean over the series:

  rings (independent RIN)   multi_ring.py, mask_drop, K = 1..32 rings; every ring has its own source noise
  rings (shared RIN)        multi_ring.py --shared-rin, K = 1..16; run r of every ring sees the same source
                            intensity noise (one laser feeds all rings), the other noise sources stay per ring
  ESN                       strong_baselines.py, 100 / 300 / 1000 nodes, spectral radius 0.95 / 1.1 / 1.25,
                            input scale 0.3, 2 random reservoirs; input noise of the ring's source referred
                            to the input
  ESN (reduced grid)        the first run, radius 0.8 / 0.95 and input scale 0.3 / 1.0, 1 reservoir
  NG-RC                     k = 16 delayed inputs spaced 4 or 6 apart, all quadratic products
  ESN, input noise 1e-3     the same ESN grid with a 3x smaller input noise (series 0 only): how much the
                            comparison depends on the input noise assumed for the software baseline

    python experiments/scaling_summary.py
"""
import csv
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
SERIES = (0, 1, 2)


def gmean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def rows(pattern):
    out = []
    for s in SERIES:
        path = os.path.join(RES, pattern.format(s))
        if os.path.exists(path):
            out += [dict(r, series=s) for r in csv.DictReader(open(path))]
    return out


def main():
    table = []

    def add(model, size, features, values):
        if len(values) == len(SERIES):
            table.append({"model": model, "size": size, "features": features, "test_nmse": gmean(values),
                          "per_series": " ".join(f"{v:.3e}" for v in values)})

    for label, pattern in (("rings (independent RIN)", "colab_scale_rings_s{}.csv"),
                           ("rings (shared RIN)", "colab_srin_rings_s{}.csv")):
        by_k = {}
        for r in rows(pattern):
            by_k.setdefault(int(r["rings"]), []).append(float(r["test_nmse"]))
        for k in sorted(by_k):
            add(label, k, 31 * k * 8, by_k[k])
    for label, pattern in (("ESN", "colab_esn_full_s{}.csv"), ("ESN (reduced grid)", "colab_scale_esn_s{}.csv")):
        by_model = {}
        for r in rows(pattern):
            by_model.setdefault(r["model"], []).append(float(r["test_nmse"]))
        for model, values in by_model.items():
            if model.startswith("ESN"):
                size = int(model.split()[1])
                add(label, size, size * 8, values)
            elif label == "ESN":
                add("NG-RC", 16, 1 + 16 + 136, values)
    # sensitivity to the assumed input noise of the software baseline (series 0 only): 1e-3 instead of 3e-3
    path = os.path.join(RES, "colab_esn_in1e3_s0.csv")
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            if r["model"].startswith("ESN"):
                size = int(r["model"].split()[1])
                table.append({"model": "ESN, input noise 1e-3 (series 0)", "size": size, "features": size * 8,
                              "test_nmse": float(r["test_nmse"]), "per_series": f"{float(r['test_nmse']):.3e}"})
    with open(os.path.join(RES, "scaling_mg.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for r in table:
        print(f"{r['model']:24s} {r['size']:5d} {r['features']:6d}  {r['test_nmse']:.2e}   ({r['per_series']})")


if __name__ == "__main__":
    main()
