"""A ring that controls another ring: nanosecond feedback on a bistable Kerr cavity.

Plant: a single-mode Kerr cavity, the one-mode limit of the same Lugiato-Lefever equation as the reservoir,
in units of the photon lifetime (time 2/kappa, detuning zeta in half-linewidths, |psi|^2 = Y the
normalised intracavity power):

    dpsi/dt = -(1 + i zeta) psi + i |psi|^2 psi + F,      steady state  F^2 = Y (1 + (zeta - Y)^2).

At zeta = 3 it is bistable for F^2 in [2.91, 5.09]; at the working pump F0 = 2 the steady states are
Y = 0.59 (lower branch), 2.0 (middle, unstable) and 3.41 (upper). One control step lasts one time unit
(about a photon lifetime, so nanoseconds for a Q of a few 10^5 to 10^6): the controller reads the drop-port
power Y (with detection noise) and sets the pump amplitude F = F0 + c for the next step, |c| <= c_max.
Noise on the plant: detuning jitter (1 % of the half-linewidth per step, as in the reservoir), pump intensity
noise, detection noise.

Controllers, all of the form "predict, then decide". From an exploratory record (random pump modulation) a
ridge readout learns the next power as an affine function of the control,
    Y[n+1] ~ A(h[n]) + B(h[n]) c[n],     features z = [x, c, c x],
where x describes the history h[n] (observed powers and the previous control); the decision is
c = clip((goal - A) / B, +-c_max). Histories x:
  linear      Y[n], Y[n-1], Y[n-2], c[n-1]         (what an analog feedback loop could compute)
  NG-RC       the last k powers and their products, c[n-1]
  ESN 31      states of a 31-node echo state network driven by Y, c[n-1]
  exact       knows the field psi and the model (no jitter): picks c from a fine grid, the upper bound
A digital ESN or NG-RC needs a converter and a processor in the loop (hundreds of nanoseconds, hundreds of
steps here); they are run without that delay as a reference for what a nonlinear predictor can do.

Tasks (``--task``):
  hold    keep Y on the unstable middle branch, Y = 2.0, with small pump changes (c_max 0.2)
  track   make Y follow a commanded trajectory across both branches (c_max 1.0)

    python experiments/ring_controls_ring.py --task track
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from combrc import fit_readout  # noqa: E402

ZETA, F0 = 3.0, 2.0
TS, DT = 1.0, 0.05
C_MAX = {"hold": 0.3, "track": 1.0}     # the exploratory record always uses the larger one
Y_GOAL_HOLD = 2.0
Y_SCALE = 4.5                       # observed powers are divided by this before driving a reservoir
NOISE = {"detuning": 0.01, "rin": 1e-3, "detect": 0.01}
TOL = 0.05
WARMUP = 200


# ---- plant ------------------------------------------------------------------------------------------

def plant_step(psi, F, zeta):
    """One control step: integrate the single-mode LLE over TS at constant pump F (arrays over trials)."""
    def rhs(p):
        return -(1 + 1j * zeta) * p + 1j * np.abs(p) ** 2 * p + F
    for _ in range(int(round(TS / DT))):
        k1 = rhs(psi)
        k2 = rhs(psi + 0.5 * DT * k1)
        k3 = rhs(psi + 0.5 * DT * k2)
        k4 = rhs(psi + DT * k3)
        psi = psi + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return psi


class Plant:
    """Several independent copies of the noisy plant."""

    def __init__(self, trials, rng, upper=None, middle=False):
        self.rng = rng
        y0 = np.where(rng.random(trials) < 0.5, 0.59, 3.41) if upper is None else np.full(trials, 3.41 if upper else 0.59)
        if middle:                                   # near the unstable middle state (the 'hold' task starts there)
            y0 = Y_GOAL_HOLD * (1 + 0.01 * rng.standard_normal(trials))
        # steady field for that power at F0: psi = F0 / (1 + i (zeta - Y))
        self.psi = F0 / (1 + 1j * (ZETA - y0))

    def observe(self):
        y = np.abs(self.psi) ** 2
        return y + NOISE["detect"] * self.rng.standard_normal(len(y))

    def step(self, c):
        zeta = ZETA + NOISE["detuning"] * self.rng.standard_normal(len(self.psi))
        F = (F0 + c) * (1 + 0.5 * NOISE["rin"] * self.rng.standard_normal(len(self.psi)))
        self.psi = plant_step(self.psi, F, zeta)

    def true_power(self):
        return np.abs(self.psi) ** 2


def steady_curve():
    """(F^2, Y) of the steady states, for a check of the bistability."""
    y = np.linspace(0, 5, 2001)
    return y * (1 + (ZETA - y) ** 2), y


# ---- exploratory record -----------------------------------------------------------------------------

def exploratory_record(n, seed, task):
    """Observed powers y[t], controls c[t] (applied after reading y[t]) under exploratory driving.

    For 'hold' half of the 60-step segments are 'middle' segments: a reference controller (the exact one)
    keeps the plant near the unstable middle state, at a goal that moves within 2.0 +- 0.4, with a random
    dither on its control. Without them the record hardly ever stays near that state (it is unstable), and a
    learned model is too coarse there. This is calibration with a stabilising reference controller, an
    idealisation: in an experiment something has to hold the device near that state while it is recorded."""
    rng = np.random.default_rng(seed)
    plant = Plant(1, rng)
    c_max = C_MAX["track"]                           # wide driving, so the record visits all three branches
    teacher = ExactController(plant, c_max) if task == "hold" else None
    modes = ["levels", "sine", "kicks", "uniform"]
    ys, cs = np.empty(n), np.empty(n)
    mode, level, amp, period, phase, goal = "levels", 0.0, 0.5, 20.0, 0.0, Y_GOAL_HOLD
    for t in range(n):
        if t % 60 == 0:
            mode = "middle" if teacher is not None and rng.random() < 0.5 else rng.choice(modes)
            amp, period, phase = rng.uniform(0.3, 1.0), 10 ** rng.uniform(0.5, 2), rng.uniform(0, 2 * np.pi)
        if mode == "levels" and t % 15 == 0:
            level = rng.uniform(-1, 1)
        if mode == "middle":
            y_obs = plant.observe()
            if t % 10 == 0:
                goal = Y_GOAL_HOLD + rng.uniform(-0.4, 0.4)
            c = float(np.clip(teacher(y_obs, None, np.array([goal]))[0] + rng.normal(0, 0.05), -c_max, c_max))
        else:                                        # same order of random draws as before the 'middle' mode existed
            c = {"levels": level, "sine": amp * np.sin(2 * np.pi * t / period + phase),
                 "kicks": rng.uniform(-1, 1) if rng.random() < 0.3 else 0.0, "uniform": rng.uniform(-1, 1)}[mode]
            c = float(np.clip(c * c_max + (rng.normal(0, 0.1 * c_max)), -c_max, c_max))
            y_obs = plant.observe()
        ys[t], cs[t] = y_obs[0], c
        plant.step(np.array([c]))
    return ys, cs


# ---- history features ---------------------------------------------------------------------------------

def lagged(y, k):
    t = np.arange(len(y))
    return np.stack([y[np.maximum(t - j, 0)] for j in range(k)], 1)


def esn_states(u, n, radius, scale, seed):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, n))
    w *= radius / np.abs(np.linalg.eigvals(w)).max()
    w_in = scale * rng.uniform(-1, 1, n)
    bias = 0.2 * rng.uniform(-1, 1, n)
    return w, w_in, bias


class History:
    """Turns the stream of observed powers (and previous controls) into a feature vector per trial."""

    def __init__(self, kind, trials, cfg=None):
        self.kind, self.cfg = kind, cfg
        self.buf = None
        if kind == "esn":
            n, radius, scale, seed = cfg
            self.w, self.w_in, self.bias = esn_states(None, n, radius, scale, seed)
            self.state = np.zeros((trials, n))

    def __call__(self, y, c_prev):
        """y, c_prev: [trials] -> features [trials, F]."""
        if self.kind == "esn":
            u = 2 * (y / Y_SCALE) - 1
            self.state = np.tanh(self.state @ self.w.T + np.outer(u, self.w_in) + self.bias)
            return np.hstack([self.state, c_prev[:, None]])
        k = 3 if self.kind == "linear" else self.cfg
        self.buf = [y] + (self.buf or [y] * k)[: k - 1]
        lags = np.stack(self.buf, 1)
        if self.kind == "linear":
            return np.hstack([lags, c_prev[:, None]])
        quad = [lags[:, i] * lags[:, j] for i in range(k) for j in range(i, k)]
        return np.hstack([lags, np.stack(quad, 1), c_prev[:, None]])


def affine(x, c):
    """[x, c, c x]: features of a readout that is affine in the control."""
    return np.hstack([x, c[:, None], c[:, None] * x])


def record_features(kind, cfg, ys, cs):
    hist = History(kind, 1, cfg)
    rows = [hist(np.array([ys[t]]), np.array([cs[t - 1] if t else 0.0]))[0] for t in range(len(ys))]
    return np.array(rows)


LOCAL = {"band": None}       # --local-band: fit the readouts only on steps whose power is within this of 2.0


def local_rows(y_now):
    """Rows used for fitting: all, or (with --local-band) those near the operating point of 'hold'."""
    return np.ones(len(y_now), bool) if LOCAL["band"] is None else np.abs(y_now - Y_GOAL_HOLD) < LOCAL["band"]


def fit_affine(x, cs, ys):
    """Readout for Y[t+1] from [x[t], c[t], c[t] x[t]]; returns the model and its validation / test NMSE."""
    keep = local_rows(ys[WARMUP:-1])
    z = torch.tensor(affine(x[WARMUP:-1], cs[WARMUP:-1])[keep])
    y = torch.tensor(ys[WARMUP + 1:][keep])
    n = len(y)
    n_train, n_val = int(0.6 * n), int(0.2 * n)
    model = fit_readout(z, y, n_train, n_val)
    pred = model.predict(z)
    err = lambda s: float(((pred[s] - y[s]) ** 2).mean() / y[s].var())
    return model, err(slice(n_train, n_train + n_val)), err(slice(n_train + n_val, None))


class RingHistory:
    """The reservoir ring driven by the observed plant power, one symbol per control step; features are the
    detected central comb lines of the last ``taps`` steps (newest first) and the previous control."""

    def __init__(self, cfg, taps, detection, trials, seed, n_modes, rk_steps):
        import closed_loop_control as clc
        self.clc = clc
        self.ring = clc.ring_with_noise(cfg, n_modes, rk_steps, detection is not None)
        self.field = self.ring.init_state(trials, seed=seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.detection, self.taps, self.buf = detection, taps, None

    def __call__(self, y, c_prev):
        from combrc import select_lines
        u = torch.tensor(np.clip(y / Y_SCALE, 0.0, 1.0)).unsqueeze(1)
        with torch.no_grad():
            spectra, self.field = self.ring(u, state=self.field, generator=self.gen)
        f = self.clc.features(select_lines(spectra[:, 0], self.clc.LINES), self.detection, 1, self.gen)
        self.buf = [f] + (self.buf or [f] * self.taps)[: self.taps - 1]
        return np.hstack([torch.cat(self.buf, -1).numpy(), c_prev[:, None]])


class AffineController:
    def __init__(self, hist, model, c_max):
        self.hist, self.model, self.c_max = hist, model, c_max

    def __call__(self, y_obs, c_prev, goal):
        x = self.hist(y_obs, c_prev)
        zeros, ones = np.zeros(len(x)), np.ones(len(x))
        a = self.model.predict(torch.tensor(affine(x, zeros))).numpy()
        b = self.model.predict(torch.tensor(affine(x, ones))).numpy() - a
        b = np.where(np.abs(b) < 1e-3, np.sign(b) * 1e-3 + (b == 0) * 1e-3, b)
        return np.clip((goal - a) / b, -self.c_max, self.c_max)


class ExactController:
    """Knows the field and the noiseless model; picks the control on a fine grid."""

    def __init__(self, plant, c_max):
        self.plant, self.grid = plant, np.linspace(-c_max, c_max, 401)

    def __call__(self, y_obs, c_prev, goal):
        psi = self.plant.psi
        trials = len(psi)
        p = plant_step(np.repeat(psi, len(self.grid)), np.tile(F0 + self.grid, trials), ZETA)
        y_next = (np.abs(p) ** 2).reshape(trials, len(self.grid))
        return self.grid[np.argmin(np.abs(y_next - np.asarray(goal)[..., None] * np.ones((trials, 1))), 1)]


class NoControl:
    def __call__(self, y_obs, c_prev, goal):
        return np.zeros(len(y_obs))


def commanded(steps):
    """Goal trajectory for 'track', across both branches: slow sine, steps, fast sine, wandering signal."""
    t = np.arange(steps)
    q = steps // 4
    levels = np.array([0.4, 3.6, 1.2, 2.8, 2.0])
    parts = [2.0 + 1.6 * np.sin(2 * np.pi * t[:q] / 200), levels[(t[:q] // max(q // 5, 1)) % 5],
             2.0 + 1.4 * np.sin(2 * np.pi * t[:q] / 23),
             2.0 + 0.8 * np.sin(2 * np.pi * t[: steps - 3 * q] / 61) + 0.6 * np.sin(2 * np.pi * t[: steps - 3 * q] / 17.3)]
    return np.concatenate(parts)


def run_loop(make_controller, task, goal, trials, seed):
    """During the WARMUP steps before t = 0 an exact controller brings the plant to the start of the goal while
    the tested controller only listens (its history fills up); from t = 0 on the tested controller acts."""
    rng = np.random.default_rng(seed)
    plant = Plant(trials, rng, middle=(task == "hold"))
    ctrl = make_controller(plant)
    c_max = C_MAX[task]
    warm = ExactController(plant, c_max)
    c = np.zeros(trials)
    misses, cs = [], []
    for t in range(-WARMUP, len(goal)):
        y_obs = plant.observe()
        g = np.full(trials, goal[max(t, 0)])
        c_new = ctrl(y_obs, c, g)
        if t < 0:
            c_new = warm(y_obs, c, g)
        plant.step(c_new)
        if t >= 0:
            misses.append(plant.true_power() - g)
            cs.append(c_new)
        c = c_new
    return np.stack(misses, 1), np.stack(cs, 1), c_max


def metrics(task, misses, cs, goal, c_max):
    hit = np.abs(misses) < TOL
    out = {"within_tolerance": hit.mean(), "rms_miss": float(np.sqrt((misses ** 2).mean())),
           "mean_control": float(np.abs(cs).mean() / c_max)}
    if task == "hold":
        late = misses[:, len(goal) // 4:]                  # after the controller had time to bring it there
        out["rms_miss_last_3_quarters"] = float(np.sqrt((late ** 2).mean()))
    else:
        out["tracking_nmse"] = float((misses ** 2).mean() / np.var(goal))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="track", choices=["hold", "track"])
    ap.add_argument("--train", type=int, default=6000, help="exploratory record length")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="only check the plant's bistability and exit")
    ap.add_argument("--skip-software", action="store_true", help="only the ring controllers")
    ap.add_argument("--groups", nargs="*", default=[], help="kinds of reservoir ring (compare_encodings groups)")
    ap.add_argument("--draws", type=int, default=32, help="random operating points per group to choose from")
    ap.add_argument("--device", default="cpu", help="where the candidate rings are simulated")
    ap.add_argument("--n-modes", type=int, default=256)
    ap.add_argument("--rk-steps", type=int, default=20)
    ap.add_argument("--taps", type=int, nargs="+", default=[3, 6])
    ap.add_argument("--name", default=None, help="output name (default ring_controls_ring_<task>_<software|ring>)")
    ap.add_argument("--esn-sizes", type=int, nargs="+", default=[31])
    ap.add_argument("--detect", type=float, default=NOISE["detect"],
                    help="rms detection noise of the plant's power reading (Y units; the branches are at 0.59 and 3.41)")
    ap.add_argument("--local-band", type=float, default=None,
                    help="fit every readout only on steps whose observed power is within this of 2.0 (for 'hold')")
    args = ap.parse_args()
    NOISE["detect"] = args.detect
    LOCAL["band"] = args.local_band
    torch.set_num_threads(2)

    if args.check:
        f2, y = steady_curve()
        turn = np.where(np.diff(np.sign(np.diff(f2))) != 0)[0] + 1
        print("turning points (F^2, Y):", [(round(f2[i], 3), round(y[i], 3)) for i in turn])
        for upper in (False, True):
            rng = np.random.default_rng(1)
            plant = Plant(1, rng, upper=upper)
            for _ in range(300):
                plant.step(np.zeros(1))
            print(f"start on {'upper' if upper else 'lower'} branch, 300 steps at F0 with noise: Y = {plant.true_power()[0]:.3f}")
        # the middle branch is unstable: start there without noise and watch it leave
        psi = np.array([F0 / (1 + 1j * (ZETA - 2.0))]) * (1 + 1e-3)
        ys = []
        for _ in range(40):
            psi = plant_step(psi, F0, ZETA)
            ys.append(abs(psi[0]) ** 2)
        print("middle branch + 0.1 % kick, Y every 5 steps:", np.round(ys[::5], 3))
        return

    task, c_max = args.task, C_MAX[args.task]
    goal = np.full(args.steps, Y_GOAL_HOLD) if task == "hold" else commanded(args.steps)
    ys, cs = exploratory_record(WARMUP + args.train, args.seed, task)
    print(f"{task}: exploratory record, Y in [{ys.min():.2f}, {ys.max():.2f}]", flush=True)

    candidates = {"linear": [("linear", None)], "NG-RC": [("ngrc", k) for k in (2, 3, 4)]}
    for n in args.esn_sizes:
        candidates[f"ESN {n}"] = [("esn", (n, r, s, e)) for r in (0.5, 0.8, 0.95) for s in (0.3, 1.0, 2.0) for e in range(3)]
    results = []

    def record(name, factory, open_nmse=float("nan")):
        import time
        t0 = time.time()
        misses, cs_loop, cm = run_loop(factory, task, goal, args.trials, seed=100 + args.seed)
        m = metrics(task, misses, cs_loop, goal, cm)
        results.append({"controller": name, "open_loop_test_nmse": open_nmse, **m})
        print(f"{name:10s} open-loop {open_nmse:9.2e}  " + "  ".join(f"{k} {v:.3g}" for k, v in m.items())
              + f"   ({time.time() - t0:.0f} s)", flush=True)

    record("no control", lambda plant: NoControl())
    record("exact", lambda plant: ExactController(plant, c_max), 0.0)
    for name, cands in ({} if args.skip_software else candidates).items():
        best = (float("inf"), None, None, None)
        for kind, cfg in cands:
            x = record_features(kind, cfg, ys, cs)
            model, val, test = fit_affine(x, cs, ys)
            if val < best[0]:
                best = (val, kind, cfg, model, test)
        _, kind, cfg, model, test = best
        record(name, lambda plant, kind=kind, cfg=cfg, model=model:
               AffineController(History(kind, args.trials, cfg), model, c_max), test)

    if args.groups:
        import closed_loop_control as clc
        import compare_encodings as ce
        from combrc import select_lines
        from combrc.evaluate import simulate
        u = torch.tensor(np.clip(ys / Y_SCALE, 0.0, 1.0))
        n = len(u) - clc.WARMUP                          # rows of the candidate lines are steps clc.WARMUP ..
        c_prev = cs[clc.WARMUP - 1: -1]

        def ring_x(lines, detection, taps):
            f = clc.features(lines, detection, taps, torch.Generator().manual_seed(0)).numpy()
            return np.hstack([f, c_prev[:, None]])

        keep = local_rows(ys[clc.WARMUP: -1])
        y_next = torch.tensor(ys[clc.WARMUP + 1:][keep])
        n_fit = len(y_next)
        n_train, n_val = int(0.6 * n_fit), int(0.2 * n_fit)
        c_now = cs[clc.WARMUP: -1]

        def fit_two_runs(x0, x1):
            """Readout fitted on run 0 (ridge strength on its validation part), scored on run 1."""
            z0, z1 = (torch.tensor(affine(x[:-1], c_now)[keep]) for x in (x0, x1))
            model = fit_readout(z0, y_next, n_train, n_val)
            pred = model.predict(z1)
            err = lambda s: float(((pred[s] - y_next[s]) ** 2).mean() / y_next[s].var())
            return model, err(slice(n_train, n_train + n_val)), err(slice(n_train + n_val, None))

        for group in args.groups:
            cfgs = [ce.sample(group, d) for d in range(args.draws)]
            all_lines = clc.candidate_lines(cfgs, u, n, args.n_modes, args.rk_steps, args.device)
            best = (float("inf"),)
            for cfg, lines in zip(cfgs, all_lines):
                if not torch.isfinite(lines).all():
                    continue
                pin = clc.pin_detection(cfg)
                for k in args.taps:
                    model, val, test = fit_two_runs(ring_x(lines[0], pin, k), ring_x(lines[1], pin, k))
                    if val < best[0]:
                        best = (val, cfg, k, model, test)
            val, cfg, k, model, test = best
            print(f"ring {group}: draw {cfg['trial']} of {args.draws}, {cfg['p_total'] * 1e3:.0f} mW, {k} taps, "
                  f"validation NMSE {val:.2e}", flush=True)
            pin = clc.pin_detection(cfg)
            record(f"ring {group}, PIN", lambda plant, cfg=cfg, k=k, model=model, pin=pin: AffineController(
                RingHistory(cfg, k, pin, args.trials, 13, args.n_modes, args.rk_steps), model, c_max), test)
            clean = select_lines(simulate(clc.ring_with_noise(cfg, args.n_modes, args.rk_steps, False), u,
                                          clc.WARMUP, n), clc.LINES)
            x_clean = ring_x(clean.reshape(-1, *clean.shape[-2:])[0], None, k)
            model_c, _, test_c = fit_two_runs(x_clean, x_clean)
            record(f"ring {group}, no noise", lambda plant, cfg=cfg, k=k, model=model_c: AffineController(
                RingHistory(cfg, k, None, args.trials, 13, args.n_modes, args.rk_steps), model_c, c_max), test_c)

    import csv
    os.makedirs("results", exist_ok=True)
    name = args.name or f"ring_controls_ring_{task}_{'ring' if args.groups else 'software'}"
    with open(f"results/{name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
