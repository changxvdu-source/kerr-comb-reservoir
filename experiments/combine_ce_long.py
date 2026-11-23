"""Combine the per-series Colab segments (one --series-seeds each) into one summary, as a single
run over the three series would aggregate them, and print it next to robust_noisy_ce.

usage: combine.py <results dir>
"""
import os
import sys

import numpy as np
import pandas as pd

d = sys.argv[1]
parts = [pd.read_csv(os.path.join(d, f"colab_robust_noisy_ce_long_s{s}_summary.csv")) for s in (0, 1, 2)]
old = pd.read_csv(os.path.join(d, "robust_noisy_ce_summary.csv")).set_index("group")


def gmean(x):
    return float(np.exp(np.mean(np.log(np.asarray(x, dtype=float)))))


rows = []
for group in parts[0]["group"]:
    p = [x.set_index("group").loc[group] for x in parts]
    rows.append({
        "task": "channel_eq", "group": group, "noise": "realistic", "n_series": 3, "test_symbols": 5000,
        "best_test_31x1": gmean([r["best_test_31x1"] for r in p]),
        "best_test_31x8": gmean([r["best_test_31x8"] for r in p]),
        "min_test": min(r["best_test_31x8"] for r in p), "max_test": max(r["best_test_31x8"] for r in p),
        "consistency": float(np.mean([r["consistency"] for r in p])),
        "ser": float(np.mean([r["ser"] for r in p])),
        "fresh_test": gmean([r["fresh_test"] for r in p]),
        "fresh_spread": float(np.mean([r["fresh_spread"] for r in p])),
        "fresh_ser": float(np.mean([r["fresh_ser"] for r in p])),
        "robust_val": gmean([r["robust_val"] for r in p]),
        "robust_changed": int(sum(r["robust_changed"] for r in p)),
        "baseline_linear": gmean([r["baseline_linear"] for r in p]),
        "baseline_esn_31x8": gmean([r["baseline_esn_31x8"] for r in p]),
        "baseline_esn_fresh": gmean([r["baseline_esn_fresh"] for r in p]),
        "baseline_esn_fresh_ser": float(np.mean([r["baseline_esn_fresh_ser"] for r in p])),
        "series0_trial": p[0]["robust_trial"],
    })
out = pd.DataFrame(rows)
out.to_csv(os.path.join(d, "colab_robust_noisy_ce_long_summary.csv"), index=False)

print("channel_eq, realistic noise, 3 series; test part 5000 symbols (old: 340). 31x8 readout, point chosen on 3 new"
      " pairs, reported on 6 new pairs. SER = symbol error rate on the new pairs, mean over 6 pairs x 3 series.")
print(f"{'':18s}{'old NMSE':>10s}{'new NMSE':>10s}{'ratio':>7s}{'old SER':>9s}{'new SER':>10s}{'pairs':>7s}")
e = out.iloc[0]
print(f"{'ESN, same size':18s}{old['baseline_esn_fresh'].iloc[0]:10.2e}{e['baseline_esn_fresh']:10.2e}"
      f"{e['baseline_esn_fresh'] / old['baseline_esn_fresh'].iloc[0]:7.2f}{'':>9s}{e['baseline_esn_fresh_ser']:10.2e}")
print(f"{'linear AR (best)':18s}{old['baseline_linear'].iloc[0]:10.2e}{e['baseline_linear']:10.2e}")
for _, r in out.iterrows():
    o = old.loc[r["group"]]
    print(f"{r['group']:18s}{o['fresh_test']:10.2e}{r['fresh_test']:10.2e}{r['fresh_test'] / o['fresh_test']:7.2f}"
          f"{o['ser']:9.4f}{r['fresh_ser']:10.2e}{r['fresh_spread']:7.0%}")
