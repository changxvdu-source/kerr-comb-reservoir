"""A physical twin as a drifting model: does a second, identical cavity on the same chip help to control the first?

Plant: the bistable single-mode Kerr cavity of ring_controls_ring.py, tracking the same commanded power, but
its detuning drifts, zeta(t) = 3 + d(t) (e.g. temperature or laser frequency wander), d a smooth random signal
of amplitude ``--amp`` and period ``--period`` steps, on top of the fast jitter. A twin cavity on the same
chip, pumped from the same laser at a fixed pump F_tw on its lower branch, sees the same d(t) (and the same
fast jitter); its power reading gives an estimate of the detuning by inverting its steady state,
    zeta_hat = Y_tw + sqrt(F_tw^2 / Y_tw - 1).
Controllers (gains chosen on separate design trials with the same drift statistics, then tested on 4 x 16 trials):
  PD + FF (nominal)   feedforward from the steady state at zeta = 3, as in ring_controls_ring_classical.py
  PID + FF (nominal)  plus integral action, the textbook answer to slow drift
  PD + FF (twin)      feedforward at the twin's estimate of zeta
  PD + FF (true)      feedforward at the true zeta (upper bound)

    python experiments/twin_control.py --period 100
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ring_controls_ring as r  # noqa: E402

F_TW = 1.5            # twin pump: below the bistable range at zeta = 3, lower branch, Y about 0.27
STEPS, WARM = 1000, 200


def drift(T, period, amp, rng):
    p1, p2 = rng.uniform(0, 2 * np.pi, 2)
    t = np.arange(T)
    return amp * (0.6 * np.sin(2 * np.pi * t / period + p1) + 0.4 * np.sin(2 * np.pi * t / (0.37 * period) + p2))


def ff(goal, zeta):
    return np.sqrt(goal * (1 + (zeta - goal) ** 2)) - r.F0


def run(kind, gains, period, amp, trials, seed):
    k1, k2, ki = gains
    rng = np.random.default_rng(seed)
    goal = r.commanded(STEPS)
    d = np.stack([drift(WARM + STEPS, period, amp, rng) for _ in range(trials)])
    plant = r.Plant(trials, rng)
    twin = np.full(trials, r.F0 * 0 + F_TW / (1 + 1j * (r.ZETA - 0.27)), dtype=complex)
    prev, integ, misses = None, np.zeros(trials), []
    c_max = r.C_MAX["track"]
    for t in range(-WARM, STEPS):
        g = goal[max(t, 0)]
        n = t + WARM
        y = plant.observe()
        y_tw = np.abs(twin) ** 2 + r.NOISE["detect"] * rng.standard_normal(trials)
        zeta_true = r.ZETA + d[:, n]
        zeta_twin = y_tw + np.sqrt(np.maximum(F_TW ** 2 / np.maximum(y_tw, 1e-3) - 1, 0))
        zeta_ff = {"nominal": np.full(trials, r.ZETA), "pid": np.full(trials, r.ZETA), "twin": zeta_twin,
                   "true": zeta_true}[kind]
        e = y - g
        de = 0 * e if prev is None else e - prev
        prev = e
        integ = np.clip(integ + e, -50, 50) if kind == "pid" else integ
        c = np.clip(-k1 * e - k2 * de - ki * integ + ff(g, zeta_ff), -c_max, c_max)
        # both cavities see the same drift and the same fast jitter (one laser, one chip); the rest is independent
        jit = r.NOISE["detuning"] * rng.standard_normal(trials)
        F = (r.F0 + c) * (1 + 0.5 * r.NOISE["rin"] * rng.standard_normal(trials))
        plant.psi = r.plant_step(plant.psi, F, zeta_true + jit)
        twin = r.plant_step(twin, F_TW * (1 + 0.5 * r.NOISE["rin"] * rng.standard_normal(trials)), zeta_true + jit)
        if t >= 0:
            misses.append(np.abs(plant.psi) ** 2 - g)
    m = np.stack(misses, 1)
    return float((m ** 2).mean() / np.var(goal)), float((np.abs(m) < r.TOL).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--period", type=float, default=100)
    ap.add_argument("--amp", type=float, default=0.3)
    ap.add_argument("--out", default="results/twin_control")
    args = ap.parse_args()
    grid_pd = [(k1, k2, 0.0) for k1 in (0.3, 0.5, 0.8, 1.2) for k2 in (-0.2, 0.0, 0.2)]
    grid_pid = [(k1, k2, ki) for k1, k2, _ in grid_pd for ki in (0.01, 0.03, 0.1, 0.3)]
    rows = []
    for kind, grid in (("nominal", grid_pd), ("pid", grid_pid), ("twin", grid_pd), ("true", grid_pd)):
        best = min(grid, key=lambda gains: run(kind, gains, args.period, args.amp, 8, seed=900)[0])
        res = [run(kind, best, args.period, args.amp, 16, seed=100 + s) for s in range(4)]
        nmse = float(np.exp(np.mean(np.log([x[0] for x in res]))))
        within = float(np.mean([x[1] for x in res]))
        rows.append({"controller": kind, "period": args.period, "amp": args.amp, "gains": " ".join(map(str, best)),
                     "tracking_nmse": nmse, "within_tolerance": within})
        print(f"period {args.period:5.0f}  {kind:8s} gains {best}  tracking NMSE {nmse:.4f}  within {within:.3f}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    path = f"{args.out}_p{args.period:g}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
