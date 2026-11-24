"""Idea B on the full comb: both directions of the mask-fed Kerr ring, differential signal, common laser noise.

Each direction of the ring carries its own intracavity field. Counter-propagating fields walk through each other
once per round trip, so each sees the other only through its round-trip-averaged power (cross-phase modulation,
factor 2):
    dE_X/dn = L E_X + i gamma L (|E_X|^2 + 2 xpm <|E_Y|^2>) E_X + sqrt(theta1) E_in,X ,   (X, Y) = (cw, ccw), (ccw, cw)
with L the ring's linear operator (loss, detuning, dispersion). No backscattering. Both directions are fed from one
laser, so the detuning jitter and the source intensity noise are common to them; the quantum field noise and the
detection noise are independent. Each direction leaves through its own drop port: 2 x 31 lines are detected.

Schemes (each from the same random operating-point distribution as the paper's mask_drop group; in the
two-direction schemes the power of the operating point is split equally between the directions):
  uni31        the paper's ring: one direction, 31 lines read                         (reference)
  uni62        the same runs, 62 lines read                                           (feature-count control)
  bi-diff      ccw carries the same mask with the signal inverted (u -> 1 - u)
  bi-one       ccw carries the same mask with a constant symbol (signal only in cw)
  bi-indep     ccw carries a second, independent random mask with the same signal
  rings-diff   bi-diff on two separate rings fed from one laser (no cross-phase)     (coupling control)

Protocol per scheme: ``--points`` random operating points on series 0 (two runs, realistic noise, cross-run
validation error of the 31 x 8 / 62 x 8 readout), the best ``--top`` re-scored on ``--pairs`` fresh pairs of runs
on series 0 to choose one, which is reported on ``--pairs`` fresh pairs on each of the series, with and without the
detuning jitter. ``--validate`` instead checks that uni31 reproduces the paper's simulator without noise.

    python experiments/counterprop_comb.py --device cuda --schemes uni31 bi-diff
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import compare_encodings as ce  # noqa: E402
from combrc import make_task, select_lines, to_features  # noqa: E402
from combrc.evaluate import WARMUP, held_out  # noqa: E402
from combrc.lle import _col, stack_rings  # noqa: E402
from combrc.readout import add_taps, detector_floor, measure  # noqa: E402

SCHEMES = ("uni31", "uni62", "bi-diff", "bi-one", "bi-indep", "rings-diff")
SIM = {"uni31": "uni", "uni62": "uni", "bi-diff": "bi-diff", "bi-one": "bi-one", "bi-indep": "bi-indep",
       "rings-diff": "rings-diff"}
TAPS = 8


def apply_mask(ring, cfg):
    """Replace the random mask of a built ring by ``cfg["mask_raw"]`` = (bias, weight), complex [K] tensors in
    arbitrary units, scaled to the operating point's side-line power exactly as compare_encodings.build scales its
    random masks. Returns the ring."""
    if "mask_raw" not in cfg:
        return ring
    bias, weight = (torch.as_tensor(x, dtype=torch.complex128) for x in cfg["mask_raw"])
    side = cfg["side_fraction"] * cfg["p_total"]
    scale = math.sqrt(side / ((bias.abs() ** 2).sum() + (weight.abs() ** 2).sum() / 12).item())
    return ring.set_input(scale * bias, scale * weight)


def line_delays(step, pattern="abs"):
    """Delay (symbols) of each input line, in the order of compare_encodings.input_lines("mask_drop").
    ``abs``: lines |mu| = 1-2 undelayed, 3-4 by ``step``, 5-6 by 2 ``step``, 7-8 by 3 ``step`` (four groups, symmetric
    about the pump); ``alt``: the four groups interleaved, mu mod 4 -> 0, 1, 2, 3 ``step``."""
    out = []
    for mu in ce.input_lines("mask_drop"):
        g = (abs(mu) - 1) // 2 if pattern == "abs" else mu % 4
        out.append(g * step)
    return out


def drive_lines(r, ut, xt):
    """Coupled-in field of stacked ring ``r`` with pump symbol ``ut`` [B] and per-line symbols ``xt`` [B, K]."""
    d = r.cw_drive(r.pump_power(ut.to(r.rdtype)))
    amp = r.input_bias + r.input_weight * (xt.to(r.rdtype) - 0.5)
    return d + _col(torch.sqrt(r.theta1)) * (amp.to(r.cdtype) @ r.input_basis)


def scheme_cfg(cfg, sim):
    return cfg if sim == "uni" else {**cfg, "p_total": cfg["p_total"] / 2}


@torch.no_grad()
def simulate(cfgs, sim, u, rows, device, seed, n_modes, steps, jitter=True, block=200, keep=62):
    """Drop-port lines of both directions, [len(cfgs) * rows, symbols after WARMUP, 2, keep] (float32, cpu).

    ``u``: [T] for every row, or [rows, T] (one input per row of each operating point). ``jitter``: bool, or a
    tensor [rows] of factors on the detuning-jitter amplitude (0 switches it off for that row)."""
    cfgs = [scheme_cfg(c, sim) for c in cfgs]
    rings = [apply_mask(ce.build(c, n_modes, steps), c) for c in cfgs]
    st = ce.with_noise(stack_rings(rings, repeats=rows).to(device), "realistic")
    if torch.is_tensor(jitter):
        jf = jitter.to(device=device, dtype=torch.float64).repeat(len(cfgs))
    else:
        jf = torch.full((len(cfgs) * rows,), 1.0 if jitter else 0.0, dtype=torch.float64, device=device)
    U = (u if u.dim() == 2 else u.unsqueeze(0).expand(rows, -1)).to(device).repeat(len(cfgs), 1)
    delayed = any(c.get("line_delays") for c in cfgs)
    if delayed:                                   # per-line delayed copies of the input, [B, T, K]
        K = len(ce.input_lines("mask_drop"))
        dl = torch.tensor([c.get("line_delays") or [0] * K for c in cfgs], device=device).repeat_interleave(rows, 0)
        idx = (torch.arange(U.shape[1], device=device)[None, :, None] - dl[:, None, :]).clamp_min(0)
        Xd = torch.gather(U.unsqueeze(-1).expand(-1, -1, K), 1, idx)
    stb = st
    if sim == "bi-indep":
        rb = [ce.build({**c, "mask_seed": 100_000 + c["trial"]}, n_modes, steps) for c in cfgs]
        stb = stack_rings(rb, repeats=rows).to(device)
    B = len(cfgs) * rows
    xpm = 0.0 if sim == "rings-diff" else 1.0
    two = sim != "uni"
    gen = torch.Generator(device=device).manual_seed(9_000_011 + 7919 * seed)
    ea = torch.cat([r.init_state(1, seed=100 + 1000 * seed + s) for r in rings for s in range(rows)]).to(device)
    eb = torch.cat([r.init_state(1, seed=200 + 1000 * seed + s) for r in rings for s in range(rows)]).to(device)
    E = torch.stack([ea, eb])
    noise = st.noise
    rho = math.exp(-1.0 / noise["detuning_corr"]) if noise["detuning_corr"] > 0 else 0.0
    drift = torch.randn(B, generator=gen, dtype=torch.float64, device=device)
    lop = st.linear_operator()
    if lop.dim() == 1:
        lop = lop.expand(B, -1)
    alpha = _col(st.alpha).reshape(-1) if st.alpha.dim() else st.alpha.expand(B)
    fn = noise["field"]
    sigma = None
    if torch.is_tensor(fn) or fn:
        sigma = (_col(fn) if torch.is_tensor(fn) else fn) * math.sqrt(st.h * st.n_modes / 2)
    h, gl = st.h, st.gamma_l

    def nonlin(F, D):
        p = F.real ** 2 + F.imag ** 2
        cross = p.mean(-1, keepdim=True).flip(0)
        return h * (1j * gl * (p + 2 * xpm * cross) * F + D)

    kept = []
    T = U.shape[1]
    for t in range(T):
        ut = U[:, t]
        if delayed:
            xt = Xd[:, t]
            da = drive_lines(st, ut, xt)
            if sim == "uni":
                db = torch.zeros_like(da)
            elif sim in ("bi-diff", "rings-diff"):
                db = drive_lines(st, 1 - ut, 1 - xt)
            elif sim == "bi-one":
                db = drive_lines(st, torch.full_like(ut, 0.5), torch.full_like(xt, 0.5))
            else:
                db = drive_lines(stb, ut, xt)
        else:
            da = st.drive(ut)
        if delayed:
            pass
        elif sim == "uni":
            db = torch.zeros_like(da)
        elif sim in ("bi-diff", "rings-diff"):
            db = st.drive(1 - ut)
        elif sim == "bi-one":
            db = st.drive(torch.full_like(ut, 0.5))
        else:
            db = stb.drive(ut)
        if da.shape[-1] == 1:
            da = da.expand(B, st.n_modes)
        if db.shape[-1] == 1:
            db = db.expand(B, st.n_modes)
        if noise["rin"]:
            s = torch.randn(B, 1, generator=gen, dtype=torch.float64, device=device)
            da, db = da * (1 + 0.5 * noise["rin"] * s), db * (1 + 0.5 * noise["rin"] * s)
        L = lop
        if noise["detuning"]:
            drift = rho * drift + math.sqrt(1 - rho ** 2) * torch.randn(B, generator=gen, dtype=torch.float64,
                                                                       device=device)
            L = lop - 1j * (noise["detuning"] * alpha * drift * jf).unsqueeze(-1)
        half = torch.exp(L * (h / 2)).unsqueeze(0)
        D = torch.stack([da, db])

        def lin(F):
            return torch.fft.ifft(half * torch.fft.fft(F))

        for _ in range(st.steps_per_symbol):
            e_ip = lin(E)
            k1 = lin(nonlin(E, D))
            k2 = nonlin(e_ip + k1 / 2, D)
            k3 = nonlin(e_ip + k2 / 2, D)
            k4 = nonlin(lin(e_ip + k3), D)
            E = lin(e_ip + k1 / 6 + k2 / 3 + k3 / 3) + k4 / 6
            if sigma is not None:
                E = E + sigma * torch.complex(torch.randn(E.shape, generator=gen, dtype=torch.float64, device=device),
                                              torch.randn(E.shape, generator=gen, dtype=torch.float64, device=device))
        if t >= WARMUP:
            a = torch.fft.fft(E) / st.n_modes
            spec = torch.fft.fftshift(_col(st.theta2) * (a.real ** 2 + a.imag ** 2), dim=-1)   # [2, B, n]
            kept.append(torch.stack([select_lines(spec[0].unsqueeze(1), keep)[:, 0],
                                     select_lines(spec[1].unsqueeze(1), keep)[:, 0]], 1).float().cpu())
        if not two:
            E[1].zero_()
    return torch.stack(kept, 1)


def feats(lines, scheme, cfg, gen):
    """Readout features [T, F] of one run's lines [T, 2, 62]."""
    sim = SIM[scheme]
    detector, shot_time = ce.detection(scheme_cfg(cfg, sim), "realistic")
    floor = detector_floor(detector)
    if scheme == "uni31":
        c = lines.shape[2] // 2 - 31 // 2          # the 31 lines centred on the pump, as select_lines(.., 31)
        p = lines[:, 0, c: c + 31].double()
    elif scheme == "uni62":
        p = lines[:, 0, :].double()
    else:
        c = lines.shape[2] // 2 - 31 // 2
        p = torch.cat([lines[:, 0, c: c + 31], lines[:, 1, c: c + 31]], 1).double()
    return add_taps(to_features(measure(p, detector, gen, shot_time=shot_time), floor=floor), TAPS)


def pair_scores(lines, scheme, cfg, y, k0=0):
    """(validation, test) NMSE of each pair of runs: fitted on one, scored on the other."""
    out = []
    for k in range(len(lines) // 2):
        gen = torch.Generator().manual_seed(1000 + k0 + k)
        xa, xb = feats(lines[2 * k], scheme, cfg, gen), feats(lines[2 * k + 1], scheme, cfg, gen)
        if not (torch.isfinite(xa).all() and torch.isfinite(xb).all()):
            out.append((float("inf"), float("inf")))
            continue
        v, te, _ = held_out(xa, y, x_test=xb, n_test=ce.N_TEST)
        out.append((v, te))
    return out


def task(seed):
    u_np, y_np = make_task("mackey_glass_h10", WARMUP + ce.N_TOTAL, seed=seed)
    return (torch.tensor(u_np, dtype=torch.float64)[: WARMUP + ce.N_TOTAL],
            torch.tensor(y_np, dtype=torch.float64)[WARMUP: WARMUP + ce.N_TOTAL])


def batched(cfgs, sim, u, rows, args, seed, jitter=True):
    per = max(1, args.max_rows // rows)
    out = []
    for i in range(0, len(cfgs), per):
        L = simulate(cfgs[i: i + per], sim, u.to(args.device), rows, args.device, seed, args.n_modes, args.steps,
                     jitter=jitter)
        out += [L[j * rows: (j + 1) * rows] for j in range(len(cfgs[i: i + per]))]
    return out


def validate(args):
    """uni31 without noise must reproduce KerrRing.forward."""
    cfg = ce.sample("mask_drop", 211)
    u, _ = task(0)
    u = u[: WARMUP + 50]
    ring = ce.build(cfg, args.n_modes, args.steps).to(args.device)
    ref, _ = ring(u.to(args.device).unsqueeze(0), state=ring.init_state(1, seed=100).to(args.device))
    ref = select_lines(ref[:, WARMUP:], 62)[0].float().cpu()
    save = dict(ce.REALISTIC)
    for k in ("rin", "detuning", "field"):
        ce.REALISTIC[k] = 0.0
    mine = simulate([cfg], "uni", u.to(args.device), 1, args.device, 0, args.n_modes, args.steps)[0, :, 0]
    ce.REALISTIC.update(save)
    err = ((mine - ref).abs().max() / ref.abs().max()).item()
    print(f"validation: max relative difference uni vs KerrRing.forward = {err:.2e}")
    return err


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schemes", nargs="+", default=list(SCHEMES), choices=SCHEMES)
    ap.add_argument("--points", type=int, default=128)
    ap.add_argument("--trial0", type=int, default=0, help="first random operating point (trial number)")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--series-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-modes", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--max-rows", type=int, default=480)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--delay-step", type=int, default=0, help="per-line delays (time-to-frequency map), see line_delays")
    ap.add_argument("--delay-pattern", default="abs", choices=["abs", "alt"])
    ap.add_argument("--out", default="results/counterprop_comb")
    args = ap.parse_args()
    if args.validate:
        validate(args)
        return
    t0 = time.time()
    u0, y0 = task(0)
    cfgs = [{k: v for k, v in ce.sample("mask_drop", t).items() if k in ce.CFG_KEYS}
            for t in range(args.trial0, args.trial0 + args.points)]
    if args.delay_step:
        cfgs = [{**c, "line_delays": line_delays(args.delay_step, args.delay_pattern)} for c in cfgs]
    sims = sorted({SIM[s] for s in args.schemes})
    stage1 = {}
    for sim in sims:
        lines = batched(cfgs, sim, u0, 2, args, seed=0)
        for s in [s for s in args.schemes if SIM[s] == sim]:
            stage1[s] = [pair_scores(L, s, c, y0)[0] for L, c in zip(lines, cfgs)]
        print(f"[{time.time() - t0:6.0f} s] stage 1 {sim}: {len(cfgs)} points", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + "_stage1.csv", "w", newline="") as f:      # the whole distribution of the random search
        w = csv.writer(f)
        w.writerow(["scheme", "trial", "val_nmse", "test_nmse"])
        for s in args.schemes:
            for c, (v, te) in zip(cfgs, stage1[s]):
                w.writerow([s, c["trial"], v, te])
    rows = []
    for s in args.schemes:
        sim = SIM[s]
        order =np.argsort([v for v, _ in stage1[s]])[: args.top]
        picks = [cfgs[i] for i in order]
        lines = batched(picks, sim, u0, 2 * args.pairs, args, seed=50)
        means = [np.mean([v for v, _ in pair_scores(L, s, c, y0, k0=50)]) for L, c in zip(lines, picks)]
        best = picks[int(np.argmin(means))]
        res, R = {}, 2 * args.pairs
        combos = [(seed, label) for seed in args.series_seeds for label in ("jitter", "no jitter")]
        U = torch.cat([task(seed)[0].unsqueeze(0).expand(R, -1) for seed, _ in combos])
        jf = torch.tensor([1.0 if label == "jitter" else 0.0 for _, label in combos for _ in range(R)])
        L = simulate([best], sim, U, len(U), args.device, 100, args.n_modes, args.steps, jitter=jf)
        for j, (seed, label) in enumerate(combos):
            y = task(seed)[1]
            res.setdefault(label, []).append(
                float(np.mean([te for _, te in pair_scores(L[j * R: (j + 1) * R], s, best, y, k0=100)])))
        a, b = ce.gmean(res["jitter"]), ce.gmean(res["no jitter"])
        row = {"scheme": s, "trial": best["trial"], "p_total_mw": best["p_total"] * 1e3, "zeta": best["zeta"],
               "q_ext1": 10 ** best["log_q_ext1"], "stage1_best_val": float(min(v for v, _ in stage1[s])),
               "test_jitter": a, "test_no_jitter": b, "jitter_penalty": a / b,
               "per_series_jitter": " ".join(f"{x:.3e}" for x in res["jitter"]),
               "per_series_no_jitter": " ".join(f"{x:.3e}" for x in res["no jitter"])}
        rows.append(row)
        print(f"[{time.time() - t0:6.0f} s] {s:10s} trial {best['trial']:3d}: test {a:.2e} with jitter, "
              f"{b:.2e} without (penalty {a / b:.2f})", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved", args.out + ".csv")


if __name__ == "__main__":
    main()
