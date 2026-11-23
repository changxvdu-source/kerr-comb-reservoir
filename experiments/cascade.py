"""Does a second ring add memory? Two rings in series, read out without delay taps.

For each task the first ring is the best single per-line-mask ring found by
compare_encodings.py (chosen on the validation part with the 31-line, no-tap readout;
its spectra come from that script's cache). Its drop-port field then drives a second
ring, all-optically (combrc/cascade.py). Random draws of the second ring vary its
coupling Q (hence photon lifetime), detuning, and the optical power it receives
(1-300 mW, set by an amplifier or attenuator between the rings).

Readouts compared, each with every requested number of delay taps:
  ring 1, 31 lines        the single ring (reference)
  ring 1, 61 lines        the single ring read out twice as wide (same feature count as "both")
  ring 2, 31 lines        only the second ring is read out
  both, 31 + 31 lines     both rings are read out
The second ring is chosen per readout on the validation part; the test part is reported,
as cross-run NMSE (readout trained on one run, tested on a run with different seed noise).

Example:
    python experiments/cascade.py --tasks narma10 mackey_glass_h10 channel_eq --workers 4
"""
import argparse
import csv
import math
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import compare_encodings as ce  # noqa: E402
from combrc import make_task, score_baselines, select_lines  # noqa: E402
from combrc.cascade import RingCascade, follower_ring  # noqa: E402
from combrc.evaluate import N_SYMBOLS, WARMUP, score_spectra  # noqa: E402
from combrc.tasks import symbol_error_rate  # noqa: E402

READOUTS = ("ring1_31", "ring1_61", "ring2_31", "both_62")


def features(lines1, lines2):
    """The four readouts' inputs, from the central 61 lines of each ring ([runs, T, 61])."""
    a, b = select_lines(lines1, 31), select_lines(lines2, 31)
    return {"ring1_31": a, "ring1_61": lines1, "ring2_31": b, "both_62": torch.cat([a, b], dim=-1)}


def score(x, y, taps, task):
    s = score_spectra(x, y, x.shape[-1], taps=taps,
                      error_rate=symbol_error_rate if task == "channel_eq" else None)
    return s["val_nmse_cross"], s["nmse_cross"], s["consistency"]


def sample_second_ring(draw):
    rng = np.random.default_rng([7, draw])
    lq1 = rng.uniform(5.8, 7.5)
    alpha, _, _ = ce.cavity(lq1)
    return {"draw": draw, "log_q_ext1": lq1, "zeta": rng.uniform(-1.0, 4.0),
            "delta0": None, "alpha": alpha, "input_power": 10 ** rng.uniform(-3.0, -0.52)}


def run(job):
    task, cfg1, second, drop_power, n_modes, steps, taps_list = job
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    u, y = (torch.tensor(a, dtype=torch.float64) for a in make_task(task, WARMUP + N_SYMBOLS))
    ring1 = ce.build(cfg1, n_modes, steps)
    ring2 = follower_ring(ring1, q_ext1=10 ** second["log_q_ext1"], delta0=second["zeta"] * second["alpha"])
    gain_db = 10 * math.log10(second["input_power"] / drop_power)
    cascade = RingCascade([ring1, ring2], gains_db=[gain_db])
    state = [torch.cat([e0, e1]) for e0, e1 in zip(cascade.init_state(1, seed=0), cascade.init_state(1, seed=1))]
    spectra, _ = cascade(u.expand(2, -1), state=state)
    spectra = spectra[:, WARMUP:]
    row = {"task": task, "draw": second["draw"], "ring2_q_total": ring2.q_total.item(),
           "ring2_lifetime_ns": (ring2.q_total / ring2.omega0).item() * 1e9, "ring2_zeta": second["zeta"],
           "ring2_input_mw": second["input_power"] * 1e3, "gain_db": gain_db}
    if not torch.isfinite(spectra).all():
        return row
    sets = features(select_lines(spectra[:, :, 0], 61), select_lines(spectra[:, :, 1], 61))
    for name in READOUTS:
        for taps in taps_list:
            val, test, cons = score(sets[name], y[WARMUP:], taps, task)
            row.update({f"val_{name}_x{taps}": val, f"test_{name}_x{taps}": test, f"cons_{name}_x{taps}": cons})
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=["narma10", "mackey_glass_h10", "channel_eq"])
    ap.add_argument("--draws", type=int, default=32, help="random second rings per task")
    ap.add_argument("--ring1-trials", type=int, default=32, help="single-ring operating points to choose ring 1 from")
    ap.add_argument("--taps", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--n-modes", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="cascade")
    ap.add_argument("--cache", default=os.path.join("results", "encodings_spectra"))
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)

    jobs, info = [], {}
    for task in args.tasks:
        _, y = make_task(task, WARMUP + N_SYMBOLS)
        y = torch.tensor(y, dtype=torch.float64)[WARMUP:]
        candidates = []
        for trial in range(args.ring1_trials):
            cfg = ce.sample("comb_mask", trial)
            lines, q_total, comb_lines = ce.spectra_of(task, cfg, args.n_modes, args.steps, args.cache, 0, "none")
            if comb_lines >= 0:
                candidates.append((score(select_lines(lines, 31), y, 1, task)[0], trial, cfg, lines, q_total))
        _, trial, cfg1, lines1, q1 = min(candidates, key=lambda c: c[0])
        drop_power = lines1.sum(-1).mean().item()
        info[task] = (trial, cfg1, q1, drop_power)
        jobs += [(task, cfg1, sample_second_ring(d), drop_power, args.n_modes, args.steps, args.taps)
                 for d in range(args.draws)]
    with Pool(args.workers) as pool:
        rows = pool.map(run, jobs, chunksize=1)

    stem = os.path.join(args.out, args.name)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (not k.startswith(("task", "draw", "ring2", "gain")), k))
    with open(stem + "_draws.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for task in args.tasks:
        trial, cfg1, q1, drop_power = info[task]
        mine = [r for r in rows if r["task"] == task and f"val_ring1_31_x{args.taps[0]}" in r]
        u, y = (torch.tensor(a, dtype=torch.float64) for a in make_task(task, WARMUP + N_SYMBOLS))
        print(f"\n=== {task}   ring 1: comb_mask trial {trial}, Q {q1:.1e}, {cfg1['p_total'] * 1e3:.0f} mW in, "
              f"{drop_power * 1e3:.2f} mW at its drop port   ({len(mine)} of {args.draws} second rings usable)")
        for taps in args.taps:
            base31 = score_baselines(u, y=y, esn_sizes=(31,), taps=taps)
            base62 = score_baselines(u, y=y, esn_sizes=(62,), taps=taps)
            linear = min(v for k, v in base31.items() if k.startswith("linear"))
            print(f"  readout taps = {taps}:   best linear AR {linear:.2e}   ESN-31 {base31['esn31']:.2e}"
                  f"   ESN-62 {base62['esn62']:.2e}")
            for name in READOUTS:
                key = f"{name}_x{taps}"
                if name.startswith("ring1"):                 # the same for every draw
                    best, note = mine[0], ""
                else:
                    best = min(mine, key=lambda r: r[f"val_{key}"])
                    median = float(np.median([r[f"test_{key}"] for r in mine]))
                    note = (f"   median over draws {median:.2e};  chosen ring 2: Q {best['ring2_q_total']:.1e} "
                            f"({best['ring2_lifetime_ns']:.1f} ns), zeta {best['ring2_zeta']:.1f}, "
                            f"{best['ring2_input_mw']:.0f} mW in ({best['gain_db']:+.0f} dB)")
                print(f"    {name:10s} test {best[f'test_{key}']:.2e}  consistency {best[f'cons_{key}']:.3f}{note}")
                summary.append({"task": task, "taps": taps, "readout": name, "test": best[f"test_{key}"],
                                "consistency": best[f"cons_{key}"], "baseline_linear": linear,
                                "baseline_esn31": base31["esn31"], "baseline_esn62": base62["esn62"],
                                "ring2_q_total": best["ring2_q_total"], "ring2_zeta": best["ring2_zeta"],
                                "ring2_input_mw": best["ring2_input_mw"]})
    with open(stem + "_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print("\nsaved", stem + "_draws.csv and _summary.csv")


if __name__ == "__main__":
    main()
