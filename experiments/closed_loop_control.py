"""Predict, then decide: a ring in a feedback loop that controls a chaotic system.

Plant: the Henon map with an additive control,
    x[n+1] = 1 - a x[n]^2 + b x[n-1] + c[n],     |c[n]| <= c_max.
Left alone it is chaotic.

Controller: the ring sees x[n] (one symbol per step) and a linear readout of its comb lines
predicts where the map would go next without control, f[n]. The decision rule is fixed and
bounded:  c[n] = clip(goal[n] - prediction, +-c_max).  When the prediction is right the next
state is the goal; the miss equals the prediction error, so prediction noise limits the loop.

Two tasks (``--task``):
  hold    keep x on the unstable fixed point x* with small kicks (c_max = 4 % of the attractor's
          width). The orbit has to come near x* by itself; from there a law that is linear in
          x[n], x[n-1] is enough.
  track   make x follow a commanded trajectory (slow and fast sine, steps, a wandering signal)
          with ample control authority. Now the controller must know the map everywhere, and
          the map is quadratic: a linear predictor cannot do it.

The readout is trained on one record of the plant under exploratory driving that covers the
regime of the task. The ring is the best of the candidate per-line masks on the validation part.

Compared in closed loop (control switched on at step 0, ``--trials`` random starts each):
  no control; exact model (upper bound); a linear predictor from x[n], x[n-1] (the local
  linearisation for "hold", the least-squares fit for "track"); an echo state network of the
  same size; the ring without noise, with realistic noise and PIN detectors, and with an optical
  pre-amplifier. The linear predictor, NG-RC and the ESN see x with input noise 3e-3 of the symbol range
  (``--input-noise``; ``--groups`` with no group runs these software baselines alone).

``--groups`` compares several kinds of ring, each chosen from ``--draws`` operating points of its group in
compare_encodings.py (e.g. comb_mask, mask_drop with stronger drop-port coupling, mask_slow2 with 2 ns
symbols, whose detector noise and shot-noise integration time follow the symbol time); the candidates are
simulated together on ``--device``. ``--esn-seeds`` gives the ESN several random reservoirs to choose from.

Examples:
    python experiments/closed_loop_control.py --task hold
    python experiments/closed_loop_control.py --task track --candidates 5 12 11 7 4
    python experiments/closed_loop_control.py --task track --groups comb_mask mask_drop mask_slow2 --draws 64 --device cuda --esn-seeds 8 --name closed_loop_track_readout
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import compare_encodings as ce  # noqa: E402
from combrc import fit_readout, nmse, select_lines, to_features  # noqa: E402
from combrc.evaluate import simulate  # noqa: E402
from combrc.readout import add_taps, detector_floor, measure  # noqa: E402

A, B = 1.4, 0.3
X_STAR = (-(1 - B) + math.sqrt((1 - B) ** 2 + 4 * A)) / (2 * A)      # 0.6314
C_MAX = {"hold": 0.1, "track": 3.0}
TOL = 0.05
LINES, WARMUP = 31, 300
NOISE = ce.REALISTIC
F64 = torch.float64


class Henon:
    """x[n+1] = 1 - a x[n]^2 + b x[n-1] + c[n]; state (x[n-1], x[n]), x observed."""
    center, scale, obs, restart, limit, x_star, seed_point = 0.0, 1.0, 1, (0.1, 0.2), 1.45, X_STAR, (0.0, 0.0)

    def free(self, p, q):
        return q, 1 - A * q ** 2 + B * p


class Ikeda:
    """The Ikeda map of a ring cavity with a nonlinear medium, z' = 1 + u z exp(i (0.4 - 6 / (1 + |z|^2))),
    with the control added to Re z; state (Re z, Im z), only Re z observed, so the predictor has to infer
    the hidden quadrature from the history."""
    center, scale, obs, restart, limit, x_star, seed_point, u = 0.6, 1.0, 0, (0.5, 0.0), 2.5, None, (0.5, 0.0), 0.9

    def free(self, p, q):
        cos, sin = (torch.cos, torch.sin) if torch.is_tensor(p) else (np.cos, np.sin)
        t = 0.4 - 6 / (1 + p * p + q * q)
        return 1 + self.u * (p * cos(t) - q * sin(t)), self.u * (p * sin(t) + q * cos(t))


PLANTS = {"henon": Henon(), "ikeda": Ikeda()}
PLANT = PLANTS["henon"]                     # set in main


def observe(state):
    return state[PLANT.obs]


def uncontrolled(state):
    """The observed value the plant would reach next without control."""
    return PLANT.free(*state)[PLANT.obs]


def advance(state, c):
    """One step of the plant with control ``c`` added to the observed quantity."""
    p, q = PLANT.free(*state)
    return (p, q + c) if PLANT.obs == 1 else (p + c, q)


def lost(f):
    """Kicked out of the basin of attraction (or too far out to be useful)."""
    return abs(f - PLANT.center) > PLANT.limit


def to_symbol(x):
    """The reading as a symbol in [0, 1]. A real modulator saturates: readings outside the range are clipped,
    which also keeps a large modulation depth from driving the pump power negative (a NaN in the ring)."""
    s = (x - PLANT.center + 1.5 * PLANT.scale) / (3.0 * PLANT.scale)
    return s.clamp(0.0, 1.0) if torch.is_tensor(s) else np.clip(s, 0.0, 1.0)


def input_noise():
    """Input noise of the linear predictor, in units of x (the symbol range is 3 scale)."""
    return 3 * PLANT.scale * NOISE["baseline_input_noise"]


def commanded(steps):
    """The trajectory to follow in the "track" task: slow sine, steps, fast sine, wandering signal."""
    t = np.arange(steps)
    q = steps // 4
    levels = np.array([-0.9, 0.3, 0.8, -0.4, 0.6])
    parts = [0.8 * np.sin(2 * np.pi * t[:q] / 200),
             levels[(t[:q] // max(q // 5, 1)) % 5],
             0.7 * np.sin(2 * np.pi * t[:q] / 23),
             0.4 * np.sin(2 * np.pi * t[: steps - 3 * q] / 61) + 0.3 * np.sin(2 * np.pi * t[: steps - 3 * q] / 17.3)
             + 0.2 * np.sin(2 * np.pi * t[: steps - 3 * q] / 7.1)]
    return PLANT.center + PLANT.scale * np.concatenate(parts)


def training_record(n, seed, task):
    """Observed values x[n] and uncontrolled next values f[n] of the plant under exploratory driving."""
    rng = np.random.default_rng(seed)
    c0, sc = PLANT.center, PLANT.scale
    state = PLANT.restart
    xs, fs = np.empty(n), np.empty(n)
    modes = ("kicks", "hold") if task == "hold" else ("kicks", "uniform", "smooth", "levels")
    mode, phase, level = "kicks", 0.0, 0.0
    amp, period, offset = 0.5, 50.0, 0.0
    for t in range(n):
        if t % 80 == 0:
            mode = modes[rng.integers(len(modes))]
            amp, period, offset = rng.uniform(0.3, 0.9), 10 ** rng.uniform(1.0, 2.5), rng.uniform(-0.3, 0.3)
            phase = rng.uniform(0, 2 * np.pi)
        f = uncontrolled(state)
        if lost(f):                                # kicked out of the basin of attraction: start again
            state = PLANT.restart
            f = uncontrolled(state)
        if mode == "kicks":
            c = sc * rng.uniform(-0.1, 0.1) if rng.random() < 0.5 else 0.0
        elif mode == "hold":
            c = float(np.clip(PLANT.x_star - f + rng.uniform(-0.03, 0.03), -C_MAX["hold"], C_MAX["hold"]))
        else:
            if mode == "uniform":
                goal = rng.uniform(-1.2, 1.2)
            elif mode == "smooth":
                goal = offset + amp * np.sin(2 * np.pi * t / period + phase) + rng.uniform(-0.2, 0.2)
            else:
                if t % 20 == 0:
                    level = rng.uniform(-1.0, 1.0)
                goal = level + rng.uniform(-0.1, 0.1)
            c = c0 + sc * float(np.clip(goal, -1.3, 1.3)) - f
        if lost(f + c):
            c = 0.0
        xs[t], fs[t] = observe(state), f
        state = advance(state, c)
    return xs, fs


def pin_detection(cfg, preamp=False):
    """Detection settings of the realistic receiver for this operating point (its symbol time)."""
    detector, shot_time = ce.detection(cfg, "realistic")
    out = {"detector_noise": detector, "shot_time": shot_time}
    return {**out, "preamp_gain_db": 10, "preamp_nf_db": 5} if preamp else out


def candidate_lines(cfgs, u, n, n_modes, steps, device, chunk=64):
    """Central comb lines [2 runs, n, LINES] of every candidate with realistic optical noise, simulated as
    stacks of rings on ``device`` (candidates that cannot share a stack go separately)."""
    from combrc.lle import stack_rings
    out, families = [None] * len(cfgs), {}
    for i, cfg in enumerate(cfgs):
        families.setdefault(ce.family(cfg), []).append(i)
    for members in families.values():
        for k in range(0, len(members), chunk):
            part = members[k: k + chunk]
            rings = [ce.build(cfgs[i], n_modes, steps) for i in part]
            state = torch.cat([r.init_state(1, seed=s) for r in rings for s in (0, 1)]).to(device)
            stack = ce.with_noise(stack_rings(rings, repeats=2).to(device), "realistic")
            gen = torch.Generator(device=device).manual_seed(17 + k)
            spectra, _ = stack(u[: WARMUP + n].to(device).expand(2 * len(part), -1), state=state, generator=gen)
            lines = select_lines(spectra[:, WARMUP:], LINES).cpu()
            for j, i in enumerate(part):
                out[i] = lines[2 * j: 2 * j + 2]
    return out


def features(lines, detection, taps, generator):
    if detection is None:
        return add_taps(to_features(lines), taps)
    floor = detector_floor(detection["detector_noise"], detection.get("preamp_gain_db", 0.0))
    return add_taps(to_features(measure(lines, generator=generator, **detection), floor=floor), taps)


def fit_predictor(x, y):
    """Readout fitted on run 0; returns it with the validation and test NMSE on run 1."""
    n_train, n_val = int(0.6 * len(y)), int(0.2 * len(y))
    model = fit_readout(x[0], y, n_train, n_val)
    pred = model.predict(x[1])
    val = nmse(y[n_train:n_train + n_val], pred[n_train:n_train + n_val]).item()
    return model, val, nmse(y[n_train + n_val:], pred[n_train + n_val:]).item()


def ring_with_noise(cfg, n_modes, steps, noisy):
    ring = ce.build(cfg, n_modes, steps)
    if noisy:
        ring.set_noise(rin=NOISE["rin"], detuning=NOISE["detuning"], detuning_corr=NOISE.get("detuning_corr", 1.0),
                       field=NOISE["field"] * ring.quantum_field_noise())
    return ring


# ---- predictors: each returns its guess of the uncontrolled next value, once per step. They are handed
# the plant state but, except for the exact model, look only at the observed x[n] ----

class NoControl:
    def __call__(self, state):
        return None


class ExactModel:
    def __call__(self, state):
        return uncontrolled(state)


class LinearPredictor:
    """w0 + w1 x[n] + w2 x[n-1], from noisy readings of x."""

    def __init__(self, weights, x_prev, seed):
        self.w, self.seen_prev = weights, x_prev.clone()
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, state):
        x = observe(state)
        seen = x + input_noise() * torch.randn(x.shape, generator=self.gen, dtype=F64)
        guess = self.w[0] + self.w[1] * seen + self.w[2] * self.seen_prev
        self.seen_prev = seen
        return guess


def quadratic_features(lags):
    """NG-RC features of [rows, k] delayed readings: a constant, the readings and their unique products."""
    k = lags.shape[1]
    quad = [lags[:, i] * lags[:, j] for i in range(k) for j in range(i, k)]
    return torch.cat([torch.ones(len(lags), 1, dtype=F64), lags, torch.stack(quad, 1)], 1)


class NgrcPredictor:
    """Next-generation reservoir computing: ridge readout over quadratic features of the k latest noisy readings."""

    def __init__(self, model, k, x_prev, seed):
        self.model, self.k = model, k
        self.hist = [x_prev.clone()] * (k - 1)
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, state):
        x = observe(state)
        seen = x + input_noise() * torch.randn(x.shape, generator=self.gen, dtype=F64)
        self.hist = [seen] + self.hist[: self.k - 1]
        return self.model.predict(quadratic_features(torch.stack(self.hist, 1)))


class Esn:
    """A tanh echo state network stepped one symbol at a time (same construction as evaluate.esn_states)."""

    def __init__(self, n_nodes, radius, scale, seed=0):
        gen = torch.Generator().manual_seed(seed)
        w = torch.randn(n_nodes, n_nodes, generator=gen, dtype=F64)
        self.w = w * (radius / torch.linalg.eigvals(w).abs().max())
        self.w_in = scale * (2 * torch.rand(n_nodes, generator=gen, dtype=F64) - 1)
        self.bias = 0.2 * (2 * torch.rand(n_nodes, generator=gen, dtype=F64) - 1)

    def step(self, state, u):
        """state [rows, nodes], u [rows] in [0, 1]."""
        return torch.tanh(state @ self.w.T + self.w_in * (2 * u.unsqueeze(-1) - 1) + self.bias)


# Online recalibration of a trained readout during the closed loop (``--adapt``). After every step the
# uncontrolled value the plant reached is known, f[n] = x[n+1] - c[n], so it serves as a fresh target:
#   bias  the intercept follows the prediction error, b <- b + rate * (f - guess)  (tracks a slow offset)
#   rls   all readout weights follow it by recursive least squares with forgetting factor ``forget``
ADAPT = {"mode": "none", "rate": 0.02, "forget": 0.999, "p0": 1e-3}


class Buffered:
    """Feeds a readout the features of the last ``taps`` steps, newest first (the layout of add_taps), and
    optionally recalibrates it online (see ADAPT); every trial row adapts on its own."""

    def __init__(self, model, taps):
        self.model, self.taps, self.buf = model, taps, None
        self.bias = self.w = self.P = self.z = self.last = None

    def predict(self, feat):
        self.buf = [feat] + (self.buf or [feat] * self.taps)[: self.taps - 1]
        x = torch.cat(self.buf, -1)
        mode = ADAPT["mode"]
        if mode == "none":
            return self.model.predict(x)
        if mode == "bias":
            if self.bias is None:
                self.bias = torch.zeros(len(x), dtype=F64)
            self.last = self.model.predict(x) + self.bias
            return self.last
        z = torch.cat([self.model.standardise(x), torch.ones(len(x), 1, dtype=F64)], -1)     # with intercept
        if self.w is None:
            w0 = torch.cat([self.model.w.reshape(-1), self.model.y_mean.reshape(1)])
            self.w = w0.expand(len(x), -1).clone()
            self.P = ADAPT["p0"] * torch.eye(z.shape[1], dtype=F64).expand(len(x), -1, -1).clone()
        self.z = z
        self.last = (z * self.w).sum(-1)
        return self.last

    def learn(self, f):
        """The uncontrolled value the plant actually reached, after the prediction ``predict`` returned."""
        mode = ADAPT["mode"]
        if mode == "bias":
            self.bias = self.bias + ADAPT["rate"] * (f - self.last)
        elif mode == "rls":
            pz = (self.P @ self.z.unsqueeze(-1))                                # [trials, d, 1]
            gain = pz[..., 0] / (ADAPT["forget"] + (self.z * pz[..., 0]).sum(-1, keepdim=True))
            self.w = self.w + gain * (f - self.last).unsqueeze(-1)
            self.P = (self.P - gain.unsqueeze(-1) * pz.transpose(1, 2)) / ADAPT["forget"]


class EsnPredictor(Buffered):
    def __init__(self, esn, model, taps, trials, seed):
        super().__init__(model, taps)
        self.esn, self.state = esn, torch.zeros(trials, len(esn.bias), dtype=F64)
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, state):
        x = observe(state)
        seen = to_symbol(x) + NOISE["baseline_input_noise"] * torch.randn(x.shape, generator=self.gen, dtype=F64)
        self.state = self.esn.step(self.state, seen)
        return self.predict(self.state)


class RingPredictor(Buffered):
    def __init__(self, ring, model, taps, detection, trials, seed):
        super().__init__(model, taps)
        self.ring, self.detection, self.field = ring, detection, ring.init_state(trials, seed=seed)
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, state):
        x = observe(state)
        spectra, self.field = self.ring(to_symbol(x).unsqueeze(1), state=self.field, generator=self.gen)
        return self.predict(features(select_lines(spectra[:, 0], LINES), self.detection, 1, self.gen))


def start_states(trials, seed):
    """Points on the free attractor, as the plant state (two tensors)."""
    rng = np.random.default_rng(seed)
    p = PLANT.seed_point[0] + rng.uniform(-0.2, 0.2, trials)
    q = PLANT.seed_point[1] + rng.uniform(-0.2, 0.2, trials)
    for _ in range(300):
        p, q = PLANT.free(p, q)
    return torch.tensor(p), torch.tensor(q)


def run_loop(predictor, goal, c_max, trials, seed):
    """Close the loop. ``goal[t]`` is where x should be after step t.

    The predictor is called from step -WARMUP on, so that its internal state has settled; control
    acts from step 0. Returns the observed states x [trials, steps], the controls c and the misses
    x[t+1] - goal[t], and the number of restarts after the plant left its basin.
    """
    state = start_states(trials, seed)
    xs, cs, misses, restarts = [], [], [], 0
    for t in range(-WARMUP, len(goal)):
        gone = lost(uncontrolled(state))           # kicked out of the basin of attraction: start again
        state = tuple(torch.where(gone, torch.full_like(v, r), v) for v, r in zip(state, PLANT.restart))
        f = uncontrolled(state)
        guess = predictor(state)
        if hasattr(predictor, "learn"):
            predictor.learn(f)                     # known once the step is done: f = x[n+1] - c[n]
        if guess is None or t < 0:
            c = torch.zeros_like(f)
        else:
            c = (goal[t] - guess).clamp(-c_max, c_max)
            c = torch.where(lost(f + c), torch.zeros_like(c), c)
        if t >= 0:
            restarts += int(gone.sum())
            xs.append(observe(state))
            cs.append(c)
            misses.append(f + c - goal[t])
        state = advance(state, c)
    return torch.stack(xs, 1), torch.stack(cs, 1), torch.stack(misses, 1), restarts


def metrics(task, cs, misses, goal, c_max):
    """Averages over the trials of one controller."""
    hit = misses.abs() < TOL                                           # [trials, steps]
    out = {"within_tolerance": hit.double().mean().item(), "rms_miss": misses.pow(2).mean().sqrt().item(),
           "mean_control": (cs.abs().mean() / c_max).item()}
    if task == "hold":
        out["rms_miss_while_held"] = misses[hit].pow(2).mean().sqrt().item() if hit.any() else float("nan")
        out["losses_per_1000_steps"] = ((hit[:, :-1] & ~hit[:, 1:]).sum(1).double().mean() / hit.shape[1] * 1000).item()
    else:
        out["tracking_nmse"] = (misses.pow(2).mean() / np.var(goal)).item()
        out["largest_miss"] = misses.abs().max().item()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="hold", choices=["hold", "track"])
    ap.add_argument("--draws", type=int, default=16, help="random per-line masks to choose the ring from")
    ap.add_argument("--candidates", type=int, nargs="+", help="choose among these draws instead of 0..draws-1")
    ap.add_argument("--train-symbols", type=int, default=2400)
    ap.add_argument("--taps", type=int, nargs="+", default=[3, 6],
                    help="readout taps to choose from (on the validation part, together with the ring)")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000, help="closed-loop steps per trial")
    ap.add_argument("--n-modes", type=int, default=256)
    ap.add_argument("--rk-steps", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--groups", nargs="*", default=["comb_mask"],
                    help="kinds of ring to compare (compare_encodings groups); none: only the software baselines")
    ap.add_argument("--input-noise", type=float, default=None,
                    help="rms input noise of the linear predictor, NG-RC and ESN as a fraction of the symbol range "
                         "(default %g: the ring's source noise referred to the input)" % NOISE["baseline_input_noise"])
    ap.add_argument("--device", default="cpu", help="where the candidate rings are simulated")
    ap.add_argument("--esn-seeds", type=int, default=1, help="random reservoirs the ESN is chosen from")
    ap.add_argument("--detuning", type=float, default=None,
                    help="rms detuning jitter of the ring in half-linewidths (default %g)" % NOISE["detuning"])
    ap.add_argument("--detuning-corr", type=float, default=None,
                    help="correlation time of the detuning jitter in symbols (default 1: white; large: slow drift)")
    ap.add_argument("--plant", default="henon", choices=list(PLANTS))
    ap.add_argument("--adapt", default="none", choices=["none", "bias", "rls"],
                    help="online recalibration of the ESN and ring readouts in the closed loop")
    ap.add_argument("--adapt-rate", type=float, default=ADAPT["rate"], help="step size of the bias correction")
    ap.add_argument("--forget", type=float, default=ADAPT["forget"], help="forgetting factor of the RLS update")
    ap.add_argument("--p0", type=float, default=ADAPT["p0"], help="initial RLS covariance (times the identity)")
    ap.add_argument("--train-seed", type=int, default=0, help="seed of the exploratory training record")
    ap.add_argument("--loop-seed", type=int, default=5, help="seed of the closed-loop start states")
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", help="output files are <out>/<name>.*; default closed_loop_<task>")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.detuning is not None:
        NOISE["detuning"] = args.detuning
    if args.detuning_corr is not None:
        NOISE["detuning_corr"] = args.detuning_corr
    if args.input_noise is not None:
        NOISE["baseline_input_noise"] = args.input_noise
    global PLANT
    PLANT = PLANTS[args.plant]
    ADAPT.update(mode=args.adapt, rate=args.adapt_rate, forget=args.forget, p0=args.p0)
    if args.task == "hold" and PLANT.x_star is None:
        ap.error("the hold task is defined for the Henon map only")
    torch.set_grad_enabled(False)
    task, c_max, n = args.task, C_MAX[args.task], args.train_symbols
    goal = np.full(args.steps, PLANT.x_star) if task == "hold" else commanded(args.steps)

    xs, fs = training_record(WARMUP + n, args.train_seed, task)
    u, y = torch.tensor(to_symbol(xs), dtype=F64), torch.tensor(fs, dtype=F64)[WARMUP:]

    # per group, the ring: best of the candidate operating points, chosen on the validation part with PIN detectors
    chosen = {}
    for group in args.groups:
        draws = list(args.candidates or range(args.draws))
        cfgs = [ce.sample(group, d) for d in draws]
        all_lines = candidate_lines(cfgs, u, n, args.n_modes, args.rk_steps, args.device)
        best = (float("inf"), None, None, None)
        for cfg, lines in zip(cfgs, all_lines):
            if not torch.isfinite(lines).all():
                continue
            pin = pin_detection(cfg)
            vals = {k: fit_predictor(features(lines, pin, k, torch.Generator().manual_seed(0)), y)[1] for k in args.taps}
            k = min(vals, key=vals.get)
            if vals[k] < best[0]:
                best = (vals[k], cfg, lines, k)
        val, cfg, noisy_lines, taps = best
        clean_lines = select_lines(simulate(ring_with_noise(cfg, args.n_modes, args.rk_steps, False), u, WARMUP, n),
                                   LINES)
        chosen[group] = (cfg, noisy_lines, clean_lines, taps)
        print(f"{group}: chosen draw {cfg['trial']} of {len(draws)}, {cfg['p_total'] * 1e3:.0f} mW, "
              f"Q {ce.build(cfg, args.n_modes, args.rk_steps).q_total.item():.1e}, {taps} readout taps, "
              f"validation NMSE {val:.2e}", flush=True)

    results, traces = [], {}

    def record(name, open_loop_nmse, predictor):
        states, cs, misses, restarts = run_loop(predictor, goal, c_max, args.trials, seed=args.loop_seed)
        entry = {"task": task, "controller": name, "open_loop_test_nmse": open_loop_nmse,
                 **metrics(task, cs, misses, goal, c_max), "restarts": restarts}
        results.append(entry)
        traces[name] = states[0].numpy()
        print(f"{name:22s} " + "   ".join(f"{k} {v:.3g}" for k, v in entry.items() if k not in ("task", "controller")),
              flush=True)

    start = start_states(args.trials, args.loop_seed)              # the loops start from these states
    x_prev0 = start[0] if PLANT.obs == 1 else observe(start)       # the reading before the first step
    record("no control", float("nan"), NoControl())
    record("exact model", 0.0, ExactModel())

    gen = torch.Generator().manual_seed(3)
    seen = torch.tensor(xs, dtype=F64) + input_noise() * torch.randn(len(xs), generator=gen, dtype=F64)
    design = torch.stack([torch.ones(len(xs) - 1, dtype=F64), seen[1:], seen[:-1]], 1)
    fit = torch.linalg.lstsq(design, torch.tensor(fs[1:], dtype=F64).unsqueeze(1)).solution[:, 0]
    if task == "hold":                                                 # the linearisation around x*
        weights = (X_STAR + 2 * A * X_STAR ** 2 - B * X_STAR, -2 * A * X_STAR, B)
        linear_nmse = float("nan")
    else:
        weights = tuple(fit.tolist())
        linear_nmse = nmse(torch.tensor(fs[1:], dtype=F64), design @ fit).item()
    record("linear predictor", linear_nmse, LinearPredictor(weights, x_prev0, seed=11))

    # NG-RC on the same noisy readings, the number of delayed readings chosen on the validation part
    seen_ngrc = torch.tensor(xs, dtype=F64) + input_noise() * torch.randn(len(xs), generator=gen, dtype=F64)
    ngrc_best = (float("inf"), None, None, None)
    for k in (2, 3, 4, 6):
        t = torch.arange(len(seen_ngrc))
        lags = torch.stack([seen_ngrc[(t - j).clamp_min(0)] for j in range(k)], 1)
        x_ngrc = quadratic_features(lags)[WARMUP:]
        model, val, test = fit_predictor(torch.stack([x_ngrc, x_ngrc]), y)
        if val < ngrc_best[0]:
            ngrc_best = (val, model, test, k)
    _, ngrc_model, ngrc_test, ngrc_k = ngrc_best
    record("NG-RC", ngrc_test, NgrcPredictor(ngrc_model, ngrc_k, x_prev0, seed=14))

    # echo state network of the same size, hyper-parameters chosen on the validation part
    u_noisy = u + NOISE["baseline_input_noise"] * torch.randn(u.shape, generator=gen, dtype=F64)
    esn_best = (float("inf"), None, None, None, None)
    for radius, scale, esn_seed in [(r, c, e) for r in (0.3, 0.5, 0.8, 0.95) for c in (0.3, 1.0, 2.0)
                                    for e in range(args.esn_seeds)]:
        esn = Esn(LINES, radius, scale, seed=esn_seed)
        s, rows = torch.zeros(1, LINES, dtype=F64), []
        for value in u_noisy:
            s = esn.step(s, value.reshape(1))
            rows.append(s[0])
        for k in args.taps:
            x_esn = add_taps(torch.stack(rows), k)[WARMUP:]
            model, val, test = fit_predictor(torch.stack([x_esn, x_esn]), y)
            if val < esn_best[0]:
                esn_best = (val, esn, model, test, k)
    _, esn, esn_model, esn_test, esn_taps = esn_best
    record(f"ESN, {LINES} nodes", esn_test, EsnPredictor(esn, esn_model, esn_taps, args.trials, seed=12))

    for group, (cfg, noisy_lines, clean_lines, taps) in chosen.items():
        label = "ring" if args.groups == ["comb_mask"] else f"ring {group}"
        for name, detection in ((f"{label}, no noise", None), (f"{label}, PIN detectors", pin_detection(cfg)),
                                (f"{label}, pre-amplifier", pin_detection(cfg, preamp=True))):
            noisy = detection is not None
            lines = noisy_lines if noisy else clean_lines
            model, _, test = fit_predictor(features(lines, detection, taps, torch.Generator().manual_seed(0)), y)
            ring = ring_with_noise(cfg, args.n_modes, args.rk_steps, noisy)
            record(name, test, RingPredictor(ring, model, taps, detection, args.trials, seed=13))

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, args.name or f"closed_loop_{task}")
    with open(stem + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    # first-trial trajectories of every controller and the goal, for figures (x[t+1] aims at goal[t])
    np.savez_compressed(stem + "_traces.npz", goal=goal, names=np.array(list(traces)),
                        traces=np.stack([traces[k] for k in traces]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shown = ["no control", "linear predictor", f"ESN, {LINES} nodes"] + [k for k in traces if k.endswith("PIN detectors")]
    span = slice(0, 600) if task == "hold" else slice(0, args.steps)
    fig, axes = plt.subplots(len(shown), 1, figsize=(10, 1.9 * len(shown)), sharex=True)
    for ax, name in zip(axes, shown):
        ax.plot(np.arange(args.steps)[span][1:], goal[span][:-1], color="gray", linewidth=0.8)   # x[t+1] aims at goal[t]
        ax.plot(np.arange(args.steps)[span], traces[name][span], ".", markersize=1.5)
        ax.set_ylabel("x")
        ax.set_title(name, fontsize=9, loc="left")
    axes[-1].set_xlabel("steps after the control is switched on (first trial; grey: the goal)")
    fig.tight_layout()
    fig.savefig(stem + ".png", dpi=130)
    print("\nsaved", stem + ".csv and .png")


if __name__ == "__main__":
    main()
