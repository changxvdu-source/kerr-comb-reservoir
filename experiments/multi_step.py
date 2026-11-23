"""Closed-loop (autoregressive) multi-step prediction.

The ring is driven by the true series while the readout is fitted; then the loop
is closed: each predicted symbol becomes the next pump power. The prediction
horizon is the number of symbols before the absolute error first exceeds
``--tol`` (the series spans [0, 1]). A closed-loop linear autoregression on the
raw input is run as a reference.

Examples:
    python experiments/multi_step.py --task mackey_glass --log-q-ext1 6.3
    python experiments/multi_step.py --task mackey_glass --ring results/train_multimode.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from combrc import KerrRing, Ridge, load_ring, make_series, select_lines, to_features  # noqa: E402


def horizon(err, tol):
    bad = (err.abs() > tol).nonzero()
    return int(bad[0]) if len(bad) else len(err)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="mackey_glass", choices=["mackey_glass", "rossler", "lorenz"])
    ap.add_argument("--ring", help=".pt file written by train_physical.py; uses that trained ring")
    ap.add_argument("--log-q-ext1", type=float, default=6.3)
    ap.add_argument("--log-q-ext2", type=float, default=7.35)
    ap.add_argument("--n-modes", type=int, default=512)
    ap.add_argument("--n-lines", type=int, default=51)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--n-train", type=int, default=1700)
    ap.add_argument("--n-predict", type=int, default=1200)
    ap.add_argument("--lam", type=float, default=1e-6)
    ap.add_argument("--ar-order", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.1)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    if args.ring:
        ring = load_ring(args.ring)
    else:
        ring = KerrRing(n_modes=args.n_modes, steps_per_symbol=args.steps,
                        q_ext1=10 ** args.log_q_ext1, q_ext2=10 ** args.log_q_ext2)

    w, n, m = args.warmup, args.n_train, args.n_predict
    u = torch.tensor(make_series(args.task, w + n + m + 1), dtype=torch.float64)
    truth = u[w + n: w + n + m]

    def feats(symbols, state):
        spectra, state = ring(symbols, state=state)
        return to_features(select_lines(spectra[0], args.n_lines)), state

    # Teacher-forced phase: symbols 0 .. w+n-1 drive the ring; fit features(t) -> u[t+1].
    x, state = feats(u[: w + n], ring.init_state(1))
    keep = torch.ones(x.shape[1], dtype=torch.bool)
    readout = Ridge(args.lam).fit(x[w:-1], u[w + 1: w + n])
    pred = []
    last = x[-1:]
    for _ in range(m):
        nxt = readout.predict(last[:, keep]).clamp(-0.5, 1.5)
        pred.append(nxt)
        last, state = feats(nxt, state)
    pred = torch.cat(pred)

    # Reference: closed-loop linear autoregression on the raw series.
    k = args.ar_order
    lagged = torch.stack([u[w - j: w - j + n - 1] for j in range(k)], 1)
    ar = Ridge(1e-10, eps=1e-12).fit(lagged, u[w + 1: w + n])
    hist = list(u[w + n - k: w + n].flip(0))
    ar_pred = []
    for _ in range(m):
        nxt = ar.predict(torch.stack(hist[:k]).unsqueeze(0)).clamp(-0.5, 1.5)[0]
        ar_pred.append(nxt)
        hist.insert(0, nxt)
    ar_pred = torch.stack(ar_pred)

    h_ring, h_ar = horizon(pred - truth, args.tol), horizon(ar_pred - truth, args.tol)
    print(f"{args.task}: horizon (|error| < {args.tol}) ring = {h_ring} symbols, "
          f"linear AR({k}) = {h_ar} symbols, out of {m}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.2))
    show = min(m, max(200, 3 * max(h_ring, h_ar)))
    ax.plot(truth[:show], "k-", lw=1, label="true")
    ax.plot(pred[:show], "r.", ms=3, label=f"ring, closed loop (horizon {h_ring})")
    ax.plot(ar_pred[:show], "b--", lw=0.8, label=f"linear AR({k}), closed loop (horizon {h_ar})")
    ax.set_xlabel("symbols after closing the loop")
    ax.set_ylabel("rescaled series")
    ax.set_ylim(-0.3, 1.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"{args.task}: autoregressive prediction")
    fig.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"multi_step_{args.task}.png")
    fig.savefig(path, dpi=130)
    print("saved", path)


if __name__ == "__main__":
    main()
