"""Score a fixed (untrained) ring while scanning one physical parameter.

For every scan value this reports, on the held-out last 20 % of the series
(protocol in combrc/evaluate.py), for the task "predict u[t + horizon]":
  * nmse          readout trained and tested on the same run,
  * nmse_cross    readout trained on one run, tested on a run with different seed noise,
  * consistency   correlation between the two runs' features (1 = reproducible, 0 = chaos),
  * comb_lines    lines within 60 dB of the strongest one,
next to input-only baselines (persistence, linear autoregression, echo state network).

Examples:
    python experiments/single_step.py --param log_q_ext1 --start 6 --stop 6.6 --num 13
    python experiments/single_step.py --horizon 10 --input-modes 8 --p-avg 0.15 --param log_q_ext1 --start 5.8 --stop 6.8 --num 11
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from combrc import KerrRing, make_series, score_baselines, score_ring  # noqa: E402
from combrc.evaluate import N_SYMBOLS, WARMUP  # noqa: E402


def ring_kwargs(param, value):
    """Translate a scan parameter into KerrRing constructor arguments."""
    if param.startswith("log_q_"):
        return {param[4:]: 10.0 ** value}
    return {param: value}


def run_point(job):
    args, value = job
    torch.set_num_threads(1)
    u = torch.tensor(make_series(args.task, WARMUP + N_SYMBOLS + args.horizon), dtype=torch.float64)
    k = args.input_modes
    kw = {"p_avg": args.p_avg, "p_swing": args.p_swing, **ring_kwargs(args.param, value)}
    ring = KerrRing(n_modes=args.n_modes, steps_per_symbol=args.steps,
                    input_modes=[m for m in range(-k, k + 1) if m], **kw)
    row = {args.param: value, "q_total": ring.q_total.item()}
    row.update(score_ring(ring, u, args.horizon, args.n_lines))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="mackey_glass", choices=["mackey_glass", "rossler", "lorenz"])
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--param", default="log_q_ext1",
                    help="KerrRing argument to scan (log_q_ext1, log_q_ext2, delta0, beta2, p_avg, p_swing, ...)")
    ap.add_argument("--start", type=float, default=6.0)
    ap.add_argument("--stop", type=float, default=6.6)
    ap.add_argument("--num", type=int, default=13)
    ap.add_argument("--p-avg", type=float, default=0.325)
    ap.add_argument("--p-swing", type=float, default=0.05)
    ap.add_argument("--input-modes", type=int, default=0,
                    help="extra input lines on each side of the pump, with random fixed weights")
    ap.add_argument("--n-modes", type=int, default=512)
    ap.add_argument("--n-lines", type=int, default=51)
    ap.add_argument("--steps", type=int, default=20, help="RK4IP steps per symbol")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    u = torch.tensor(make_series(args.task, WARMUP + N_SYMBOLS + args.horizon), dtype=torch.float64)
    base = score_baselines(u, args.horizon)
    print("input-only baselines (test NMSE):", {k: f"{v:.2e}" for k, v in base.items()})

    values = np.linspace(args.start, args.stop, args.num)
    with Pool(args.workers) as pool:
        rows = pool.map(run_point, [(args, float(v)) for v in values])

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.task}_h{args.horizon}_in{args.input_modes}_{args.param}"
    stem = os.path.join(args.out, f"scan_{tag}")
    with open(stem + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print("  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax1.semilogy(values, [r["nmse"] for r in rows], "o-", label="ring, same run")
    ax1.semilogy(values, [r["nmse_cross"] for r in rows], "s--", label="ring, tested on a different noise seed")
    best_ar = min(v for k, v in base.items() if k.startswith("linear"))
    for label, value, style in (("best linear AR", best_ar, "--"), ("ESN, 51 nodes", base["esn51"], ":")):
        ax1.axhline(value, color="gray", linestyle=style, label=f"baseline: {label}")
    ax1.set_ylabel(f"test NMSE, {args.horizon} step(s) ahead")
    ax1.legend(fontsize=8)
    ax1.set_title(f"{args.task}: fixed ring vs {args.param}")
    ax2.plot(values, [r["consistency"] for r in rows], "o-", color="C3")
    ax2.set_ylabel("consistency (2 noise seeds)")
    ax2.set_xlabel(args.param)
    fig.tight_layout()
    fig.savefig(stem + ".png", dpi=130)
    print("saved", stem + ".csv", "and .png")


if __name__ == "__main__":
    main()
