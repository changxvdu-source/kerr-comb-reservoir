"""The ring as a sensor: estimate a detuning shift from its comb lines, with the pump power drifting.

What hurts the reservoir (its sensitivity to detuning) is what a microring sensor measures: a refractive-index
or temperature change shifts the resonance, i.e. the detuning. Question: at the same device and optical
power, does the coherently seeded comb (input comb with the spectral-phase mask, Kerr mixing) with a
readout over many lines estimate the shift better than reading the pump line alone, the classic approach?

Each measurement: the ring (best Mackey-Glass operating point, trial 211 of mask_drop, static input, u = 0.5)
starts empty, settles for ``--settle`` symbols at a detuning shift Delta (uniform in +-``--range``
half-linewidths) and a pump power factor p (uniform in [1 - ``--drift``, 1 + ``--drift``], unknown to the
readout), and the detected lines are averaged over ``--window`` symbols. Realistic noise: laser of
``--linewidth`` white frequency noise (as detuning jitter on top of Delta), intensity noise, quantum field
noise, shot noise, 0.3 uW detectors. Two devices at the same total power:
  comb    the operating point as it is (pump + 16 masked input lines)
  pump    the same ring with all the power in the pump (no input lines): the classic CW-pumped resonator
Readouts (ridge, strength chosen on a validation part; samples: --train / --val / --test):
  R1 pump line        polynomial (degree 5) of the log pump-line power
  R2 pump + ref       polynomial (degree 3) of the log pump-line power and a reference reading of the input
                      power (a tap photodiode, 0.1 % noise)
  R3 31 lines         linear in the log powers of the 31 central lines
  R4 31 lines + ref   the same plus the reference reading
Score: rms error of the estimated Delta on the test part, over the whole range and for |Delta| < 0.5.

    python experiments/ring_sensor.py --device cuda
"""
import argparse
import csv
import itertools
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import compare_encodings as ce  # noqa: E402
from laser_noise import NU, laser_frequency  # noqa: E402
from combrc import fit_readout, select_lines  # noqa: E402
from combrc.lle import stack_rings  # noqa: E402
from combrc.readout import detector_floor, measure  # noqa: E402

LINES = 31
PUMP = LINES // 2


@torch.no_grad()
def measure_rings(cfg, delta, power, nu, settle, window, n_modes, steps, device, seed, chunk=256):
    """Averaged detected central-line powers [rows, LINES] for detuning shifts ``delta`` (half-linewidths),
    pump power factors ``power`` and laser frequency records ``nu`` ([rows, settle + window], Hz)."""
    ring = ce.build(cfg, n_modes, steps)
    hwhm = NU / (2 * float(ring.q_total))
    detector, shot_time = ce.detection(cfg, "realistic")
    floor = detector_floor(detector)
    gen_det = torch.Generator().manual_seed(seed)
    rows, T = len(delta), settle + window
    out = []
    for k in range(0, rows, chunk):
        sl = slice(k, min(k + chunk, rows))
        n = sl.stop - sl.start
        stack = ce.with_noise(stack_rings([ring], repeats=n).to(device), "realistic")
        state = torch.cat([ring.init_state(1, seed=10_000 + seed * 100_000 + k + j) for j in range(n)]).to(device)
        gen = torch.Generator(device=device).manual_seed(7_000_003 + seed * 1009 + k)
        det = torch.tensor(delta[sl, None] + nu[sl] / hwhm, dtype=torch.float64)
        amp = np.sqrt(power[sl])                       # the drive is a field: power factor p -> amplitude sqrt(p)
        rin = torch.tensor((amp[:, None] - 1) / (0.5 * ce.REALISTIC["rin"])
                           + np.random.default_rng(seed + k).standard_normal((n, T)), dtype=torch.float64)
        u = torch.full((n, T), 0.5, dtype=torch.float64, device=device)
        spectra, _ = stack(u, state=state, generator=gen, detuning=det.to(device), rin=rin.to(device))
        lines = select_lines(spectra[:, settle:], LINES).cpu()
        read = measure(lines, detector, gen_det, shot_time=shot_time).mean(1)
        out.append(torch.log10(read + floor))
        del stack, spectra, state
    return torch.cat(out).numpy()


def poly(cols, degree):
    """All monomials of the columns up to ``degree`` (no constant)."""
    feats = []
    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(cols.shape[1]), d):
            feats.append(np.prod(cols[:, combo], axis=1))
    return np.stack(feats, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="mask_drop")
    ap.add_argument("--trial", type=int, default=211)
    ap.add_argument("--range", type=float, default=2.0)
    ap.add_argument("--drift", type=float, default=0.1)
    ap.add_argument("--linewidth", type=float, default=1.6e3)
    ap.add_argument("--settle", type=int, default=300)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--train", type=int, default=1024)
    ap.add_argument("--val", type=int, default=256)
    ap.add_argument("--test", type=int, default=512)
    ap.add_argument("--devices", nargs="+", default=["comb", "pump"])
    ap.add_argument("--n-modes", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ring_sensor")
    args = ap.parse_args()
    torch.set_num_threads(4)
    ce.REALISTIC["detuning"] = 1.0                     # the supplied sequences are the detuning itself
    rng = np.random.default_rng(args.seed)
    rows = args.train + args.val + args.test
    delta = rng.uniform(-args.range, args.range, rows)
    power = rng.uniform(1 - args.drift, 1 + args.drift, rows)
    ref = np.log10(power * (1 + 1e-3 * rng.standard_normal(rows)))
    nu = laser_frequency(rows, args.settle + args.window, 1e-9, args.linewidth, 0.0, 0.0, 2 ** 12, seed=args.seed + 77)
    base = ce.sample(args.group, args.trial)
    results = []
    saved = {"delta": delta, "power": power, "ref": ref}
    for device_kind in args.devices:
        cfg = {"comb": dict(base), "pump": dict(base, side_fraction=0.0),
               "pump_low": dict(base, side_fraction=0.0, p_total=0.1 * base["p_total"]),
               "comb_low": dict(base, p_total=0.1 * base["p_total"])}[device_kind]
        logp = measure_rings(cfg, delta, power, nu, args.settle, args.window, args.n_modes, args.steps, args.device,
                             args.seed)
        saved[f"logp_{device_kind}"] = logp
        z = (logp - logp[: args.train].mean(0)) / (logp[: args.train].std(0) + 1e-9)
        r = (ref - ref[: args.train].mean()) / ref[: args.train].std()
        pr = np.stack([z[:, PUMP], r], 1)
        readouts = {"R1 pump line": poly(z[:, [PUMP]], 5),
                    "R2 pump + ref": poly(pr, 3),
                    "R2b pump + ref, degree 6": poly(pr, 6),
                    "R2c pump + ref, nearest neighbours": "knn",
                    "R3 31 lines": z,
                    "R4 31 lines + ref": np.hstack([z, r[:, None]])}
        near = np.abs(delta[-args.test:]) < 0.5
        for name, x in readouts.items():
            if isinstance(x, str):          # no functional form at all: mean Delta of the k nearest training points
                fit = slice(0, args.train + args.val)
                d2 = ((pr[-args.test:, None, :] - pr[None, fit, :]) ** 2).sum(-1)
                best_k, best_err = None, float("inf")
                for k in (3, 5, 10, 20):    # k chosen on the validation part
                    dv = ((pr[args.train: args.train + args.val, None, :] - pr[None, : args.train, :]) ** 2).sum(-1)
                    pv = delta[: args.train][np.argsort(dv, 1)[:, :k]].mean(1)
                    e = np.sqrt(((pv - delta[args.train: args.train + args.val]) ** 2).mean())
                    if e < best_err:
                        best_k, best_err = k, e
                pred = delta[fit][np.argsort(d2, 1)[:, :best_k]].mean(1)
            else:
                model = fit_readout(torch.tensor(x), torch.tensor(delta), args.train, args.val)
                pred = model.predict(torch.tensor(x[-args.test:])).numpy()
            err = pred - delta[-args.test:]
            entry = {"device": device_kind, "readout": name, "rms_all": float(np.sqrt((err ** 2).mean())),
                     "rms_near": float(np.sqrt((err[near] ** 2).mean())), "range": args.range, "drift": args.drift,
                     "linewidth": args.linewidth, "window": args.window}
            results.append(entry)
            print(f"{device_kind:5s} {name:18s} rms error {entry['rms_all']:.4f} half-linewidths "
                  f"(|Delta| < 0.5: {entry['rms_near']:.4f})", flush=True)
        pump_span = logp[:, PUMP].max() - logp[:, PUMP].min()
        print(f"   {device_kind}: pump line spans {pump_span:.2f} decades; lines above the floor: "
              f"{int((logp.mean(0) > math.log10(detector_floor(ce.detection(cfg, 'realistic')[0])) + 0.5).sum())}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out + "_features.npz", **saved)
    with open(args.out + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
