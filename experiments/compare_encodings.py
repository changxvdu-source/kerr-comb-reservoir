"""Which way of feeding the signal into a Kerr ring computes best? Equal-budget random search.

Encodings (none of them trained by gradients here):
  power, detuning, phase   one CW pump whose power / detuning / phase carries the signal
                           (Shishavan 2025 / Cuevas 2025 / Arabieh 2026), cavity starts empty;
  *_soliton                the same three, started from a single soliton at an operating
                           point drawn inside the soliton existence range;
  comb_uniform             pump plus 16 input lines of equal amplitude and phase, all
                           amplitude-modulated together (a modulated pulse train);
  comb_mask                pump plus 16 input lines with independent random complex bias
                           and modulation weight per line (per-line input mask).

Every encoding gets the same number of random operating points, drawn from the same
ranges of total optical input power (10-300 mW; solitons 30-300 mW) and coupling Q.
Per task, series and encoding the operating point is chosen on the validation part of
the series and reported on the test part, using the cross-run error (readout trained
on one run, tested on a run with independent noise). With several ``--series-seeds``
the reported number is the geometric mean over the series, with its min..max range.

``--noise realistic`` switches on laser intensity noise, detuning jitter and
quantum-level field noise in the simulation, photon shot noise and detector noise in the readout; the
baselines then see inputs with an approximately equivalent amount of noise.

Readouts compared ("lines x taps"): the readout sees the central 31 or 61 comb lines
of the current symbol, and with taps > 1 also those of the previous taps - 1 symbols
(readout memory). Baselines get the same memory: the echo state network's readout
sees the same number of past states, and the linear autoregression uses up to 32 lags.

The simulated spectra (central 61 lines of both runs) are cached under
``<out>/<name>_spectra`` so that new readout variants can be scored without
re-simulating; the folder can be deleted at any time.

Examples:
    python experiments/compare_encodings.py --series-seeds 0 1 2 --workers 16
    python experiments/compare_encodings.py --series-seeds 0 1 2 --noise realistic --name encodings_noisy --cache results/encodings_spectra
    python experiments/compare_encodings.py --tasks narma10 --variants 31x8 31x12 31x16 --main 31x12 --name encodings_narma_taps --cache results/encodings_spectra
"""
import argparse
import csv
import math
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from combrc import KerrRing, make_task, score_baselines, select_lines, stack_rings  # noqa: E402
from combrc import fit_readout, nmse  # noqa: E402
from combrc.evaluate import N_SYMBOLS, WARMUP, _splits, esn_states, score_spectra, simulate  # noqa: E402
from combrc.readout import add_taps  # noqa: E402
from combrc.tasks import is_symbol_task, symbol_error_rate  # noqa: E402

GROUPS = ("power", "detuning", "phase", "power_soliton", "detuning_soliton", "phase_soliton",
          "comb_uniform", "comb_mask")
# Ablations of the per-line mask: operating points from the same ranges of Q, power, detuning and depth as
# the comb groups, but a different number of signal-carrying lines, or less freedom in their amplitudes.
# Value: number of lines.
ABLATIONS = {"comb_mask4": 4, "comb_mask8": 8, "comb_mask32": 32,    # the full mask on fewer or more lines
             "comb_static": 16,       # random spectrum, one common modulation (weight = depth * bias)
             "comb_phase": 16,        # equal magnitudes, random phases of bias and weight
             "comb_amplitude": 16,    # random magnitudes, all lines in phase
             "comb_mask64": 64}       # the full mask on 64 lines (appended last: keeps the draws of the others)
# Sources of memory beyond the photon lifetime, on top of the per-line mask (opt-in; compare them with
# comb_mask under several readout taps, e.g. --tasks narma10 --variants 31x1 31x4 31x8 31x12 --main 31x12):
#   mask_fast2, mask_fast4   symbols 2 or 4 times shorter (0.5, 0.25 ns): the ring remembers more symbols
#   mask_delay               the input lines also carry the signal delayed by 1..12 symbols (delay line)
#   mask_loop                drop-port output fed back to the input through a one-symbol fibre loop
MEMORY = {"mask_fast2": 0.5e-9, "mask_fast4": 0.25e-9, "mask_delay": 1e-9, "mask_loop": 1e-9}   # symbol time
# Ways to move the noise floor, on top of the per-line mask (opt-in; Mackey-Glass with --noise realistic, scored
# once with ideal detectors, --detector 0, and once with the usual 0.3 uW):
#   mask_sin     a Si3N4 ring (group index 2.1, gamma 1 /(W m), beta2 -6e-26 s^2/m) with the same Q values; the
#                power range is scaled down by the ratio of gamma * L, so the normalised pump spans the same range
#                with about 40 times fewer photons
#   mask_bright  the silica ring at 10 times the power range (0.1-3 W): more photons per unit of nonlinearity
#   mask_drop    drop-port coupling Q drawn from 1e6..2.2e7 instead of the fixed 2.2e7: more light at the detectors
#   mask_slow2   2 ns symbols: twice the integration time per reading, half the detector bandwidth
SILICA = {"n_group": 1.45, "gamma": 1.6e-2, "beta2": -4.0e-27}
SIN = {"n_group": 2.1, "gamma": 1.0, "beta2": -6.0e-26}
LQ2 = math.log10(2.2222e7)            # default drop-port coupling Q
FLOOR = {"mask_sin": 1e-9, "mask_bright": 1e-9, "mask_drop": 1e-9, "mask_slow2": 2e-9}
# A ring that keeps pace with a 28 GBd link (opt-in; used with the fibre_pam4 tasks): symbols of 1/28 GHz,
# so the cavity must answer within tens of picoseconds: coupling Q 1e4..2e5 (photon lifetime ~10-150 ps),
# drop-port coupling as strong or up to 3 times weaker, total input power 10 mW .. 1 W. At such low Q the
# Kerr effect is weak at any practical power (a normalised pump F^2 of order 1 would need tens of watts in
# Si3N4 and kilowatts in silica), so these rings work close to linear and the nonlinearity comes mostly from
# square-law detection. Detector noise and shot-noise time follow the symbol time.
LINK = {"link_sin": 1 / 28e9, "link_silica": 1 / 28e9}
# mask_delay_drop: the delay line of mask_delay with the stronger drop-port coupling of mask_drop
# (appended last, so that the random draws of every earlier group stay what they were)
COMBINED = {"mask_delay_drop": 1e-9}
MASKED = {**MEMORY, **FLOOR, **LINK, **COMBINED}  # groups built on the per-line mask; value: symbol time
# Linear-cavity controls (opt-in): the same operating points, masks and readout as the named group, but with
# the Kerr term switched off (gamma = 0). Whatever these still achieve comes from the interference of the input
# lines, square-law detection, the logarithm and the cavity's linear filtering, not from four-wave mixing.
LINEAR = {"lin_comb_mask": "comb_mask", "lin_comb_uniform": "comb_uniform"}
ALL_GROUPS = (GROUPS + tuple(ABLATIONS) + tuple(MEMORY) + tuple(FLOOR) + tuple(LINK) + tuple(COMBINED)
              + tuple(LINEAR))
TASKS = ("mackey_glass_h10", "narma10", "channel_eq")
COMB_LINES = [m for m in range(-8, 9) if m]
CACHED_LINES = 61
VARIANTS = ((31, 1), (31, 4), (31, 8), (61, 1))      # default (comb lines, readout taps)
MAIN = (31, 8)                                       # default variant for selection details and the re-check
# Scored symbols per series: N_SYMBOLS split 60/20/20, or with --test-symbols the training and validation
# parts of N_SYMBOLS followed by a test part of N_TEST symbols. Set in every process by set_length.
N_TEST, N_TOTAL = None, N_SYMBOLS


def set_length(n_test, detector_uw=None, fit_symbols=None):
    """Per-process settings: the length of the test part, optionally the detector noise (uW rms), and
    optionally the length of the training + validation part (default 0.8 N_SYMBOLS; split 3:1)."""
    global N_TEST, N_TOTAL
    if detector_uw is not None:
        REALISTIC["detector"] = detector_uw * 1e-6
    N_TEST = n_test or None
    if fit_symbols and N_TEST is None:
        raise ValueError("a longer training part needs an explicit test length")
    N_TOTAL = N_SYMBOLS if N_TEST is None else int(fit_symbols or 0.8 * N_SYMBOLS) + N_TEST
# "realistic" noise: 0.1 % rms intensity noise, detuning jitter of 1 % of the half-linewidth, field noise
# at the vacuum level, 0.3 uW rms per detected line; baselines get input noise rin / (modulation depth 0.33).
# 0.3 uW is mid-range for amplified InGaAs PIN receivers of about 1 GHz bandwidth: datasheet NEPs of
# 5.6-20 pW/sqrt(Hz) (Thorlabs FPD610 5.6, Femto HSPR-X-I-2G-IN 11, Thorlabs FPD310 14, Newport 1611 20)
# over a noise bandwidth of 0.5-1 GHz give 0.13-0.63 uW. Photon shot noise is that of one reading per
# symbol (1 ns integration).
REALISTIC = {"rin": 1e-3, "detuning": 0.01, "field": 1.0, "detector": 3e-7, "shot_time": 1e-9,
             "baseline_input_noise": 3e-3}
# Noise inside the simulation, by cache tag; the single-source tags are used by noise_sources.py.
OPTICAL_NOISE = {"none": (), "realistic": ("rin", "detuning", "field"),
                 "rin": ("rin",), "detuning": ("detuning",), "field": ("field",)}
CFG_KEYS = ("group", "trial", "log_q_ext1", "p_total", "zeta", "delta0", "depth", "side_fraction",
            "symbol_time", "delay", "loop_power", "loop_phase", "log_q_ext2", "n_group", "gamma", "beta2")


def detection(cfg, noise):
    """(detector noise, shot-noise integration time) of one reading per symbol. The detector noise is
    the realistic value for 1 ns symbols, scaled with the square root of the bandwidth for shorter ones."""
    if noise != "realistic":
        return 0.0, 0.0
    t = cfg.get("symbol_time", 1e-9)
    return REALISTIC["detector"] * math.sqrt(1e-9 / t), t


def delayed(u, cfg):
    """The signal delayed by the operating point's delay line (the first symbol repeated before the start);
    ``u`` itself when it has none."""
    d = int(cfg.get("delay", 0))
    return u if d == 0 else torch.cat([u[:1].expand(d), u[:-d]])


def tag(variant):
    return f"{variant[0]}x{variant[1]}"


def cavity(lq1, lq2=LQ2, material=None):
    """(alpha, theta1, gamma*L) of a ring with these couplings (silica unless ``material`` says otherwise)."""
    ring = KerrRing(n_modes=8, q_ext1=10 ** lq1, q_ext2=10 ** lq2, **(material or SILICA))
    return ring.alpha.item(), ring.theta1.item(), ring.gamma_l


def input_lines(group):
    """Mode numbers (relative to the pump) of the comb lines that carry the signal; none for pump-only groups."""
    group = LINEAR.get(group, group)
    if not (group.startswith("comb") or group in MASKED):
        return []
    half = ABLATIONS.get(group, len(COMB_LINES)) // 2
    return [m for m in range(-half, half + 1) if m]


def sample(group, trial):
    """Draw one operating point. Deterministic in (group, trial), so every task sees the same points."""
    if group in LINEAR:                   # the very same point and mask as the Kerr group, without the Kerr term
        return {**sample(LINEAR[group], trial), "group": group, "gamma": 0.0}
    rng = np.random.default_rng([ALL_GROUPS.index(group), trial])
    soliton = group.endswith("_soliton")
    encoding = group.split("_")[0]
    while True:
        lq1 = rng.uniform(6.5, 7.2) if soliton else rng.uniform(5.8, 7.2)
        power = 10 ** (rng.uniform(-1.5, -0.52) if soliton else rng.uniform(-2.0, -0.52))
        alpha, theta1, gl = cavity(lq1)
        if not soliton:
            zeta = rng.uniform(-1.0, 4.0)
            break
        zeta_max = 0.9 * math.pi ** 2 / 8 * gl * theta1 * power / alpha ** 3   # soliton existence limit
        if zeta_max > 2.5:
            zeta = rng.uniform(2.0, zeta_max)
            break
    cfg = {"group": group, "trial": trial, "log_q_ext1": lq1, "p_total": power,
           "zeta": zeta, "delta0": zeta * alpha, "depth": 0.0, "side_fraction": 0.0,
           "symbol_time": MASKED.get(group, 1e-9), "delay": 0, "loop_power": 0.0, "loop_phase": 0.0,
           "log_q_ext2": LQ2, **SILICA}
    if group in MASKED:                   # the per-line mask, plus a source of memory or a change of platform
        cfg["depth"] = rng.uniform(0.05, 0.9)
        cfg["side_fraction"] = rng.uniform(0.1, 0.9)
        if group in ("mask_delay", "mask_delay_drop"):
            cfg["delay"] = int(rng.integers(1, 13))
            if group == "mask_delay_drop":
                cfg["log_q_ext2"] = rng.uniform(6.0, LQ2)
                cfg["delta0"] = zeta * cavity(lq1, cfg["log_q_ext2"])[0]
        elif group == "mask_loop":        # loop power transmission 0.3 % .. 80 %, any loop phase
            cfg["loop_power"] = 10 ** rng.uniform(-2.5, -0.1)
            cfg["loop_phase"] = rng.uniform(0, 2 * math.pi)
        elif group == "mask_sin":         # same Q values, so the same alpha; fewer watts for the same F^2
            cfg.update(SIN)
            cfg["p_total"] = power * gl / cavity(lq1, material=SIN)[2]
        elif group == "mask_bright":
            cfg["p_total"] = 10 * power
        elif group == "mask_drop":
            cfg["log_q_ext2"] = rng.uniform(6.0, LQ2)
            cfg["delta0"] = zeta * cavity(lq1, cfg["log_q_ext2"])[0]
        elif group in LINK:
            material = SIN if group == "link_sin" else SILICA
            lq1 = rng.uniform(4.0, 5.3)
            lq2 = lq1 + rng.uniform(0.0, 0.5)
            alpha, theta1, gl = cavity(lq1, lq2, material)
            power = 10 ** rng.uniform(-2.0, 0.0)      # 10 mW .. 1 W: what a chip and its input can take
            cfg.update(material)
            cfg.update(log_q_ext1=lq1, log_q_ext2=lq2, p_total=power, delta0=zeta * alpha)
        return cfg
    if encoding == "power":
        cfg["depth"] = rng.uniform(0.02, 0.5) if soliton else rng.uniform(0.05, 1.0)
    elif encoding in ("detuning", "phase"):
        cfg["depth"] = 10 ** rng.uniform(-1.3, 0.0 if soliton and encoding == "detuning" else 0.5)
    else:
        cfg["depth"] = rng.uniform(0.05, 0.9)
        cfg["side_fraction"] = rng.uniform(0.1, 0.9)
    return cfg


def build(cfg, n_modes, steps):
    if cfg["group"] in LINEAR:            # built like its Kerr group; cfg carries gamma = 0
        return build({**cfg, "group": LINEAR[cfg["group"]]}, n_modes, steps)
    group, power, depth = cfg["group"], cfg["p_total"], cfg["depth"]
    encoding = "comb" if group in MASKED else group.split("_")[0]
    symbol_time = cfg.get("symbol_time", 1e-9)
    # the same RK step length as for 1 ns symbols: fewer steps per shorter symbol
    # (at least 8 steps per symbol for symbols much shorter than 1 ns, where few round trips fit in a symbol)
    kw = {"n_modes": n_modes, "steps_per_symbol": max(8 if symbol_time < 0.2e-9 else 1, round(steps * symbol_time / 1e-9)),
          "symbol_time": symbol_time, "q_ext1": 10 ** cfg["log_q_ext1"], "q_ext2": 10 ** cfg.get("log_q_ext2", LQ2),
          **{k: cfg.get(k, v) for k, v in SILICA.items()},
          "delta0": cfg["delta0"], "p_avg": power, "p_swing": 0.0}
    if encoding == "power":
        kw["p_swing"] = depth * power
    elif encoding == "detuning":
        kw["detuning_depth"] = depth
    elif encoding == "phase":
        kw["phase_depth"] = depth
    else:
        side = cfg["side_fraction"] * power
        kw.update(p_avg=power - side, p_swing=2 * depth * (power - side), input_modes=input_lines(group))
    ring = KerrRing(**kw)
    if encoding == "comb":
        k = len(input_lines(group))
        if group == "comb_uniform":       # amplitude factor (1 + depth * (u - 0.5)) on every line
            bias = torch.full((k,), math.sqrt(side / k), dtype=torch.complex128)
            weight = depth * bias
        else:                             # ``mask_seed`` lets several operating points share one mask
            gen = torch.Generator().manual_seed(cfg.get("mask_seed", cfg["trial"]))
            bias, weight = (torch.complex(*torch.randn(2, k, generator=gen, dtype=torch.float64)) for _ in range(2))
            if group == "comb_static":
                weight = depth * bias
            elif group == "comb_phase":
                bias, weight = bias / bias.abs(), weight / weight.abs()
            elif group == "comb_amplitude":
                bias, weight = bias.abs().to(torch.complex128), weight.abs().to(torch.complex128)
            later = torch.zeros(k, dtype=torch.complex128)
            if cfg.get("delay", 0):           # a third random weight per line for the delayed signal
                later = torch.complex(*torch.randn(2, k, generator=gen, dtype=torch.float64))
            scale = math.sqrt(side / ((bias.abs() ** 2).sum() + (weight.abs() ** 2).sum() / 12
                                      + (later.abs() ** 2).sum() / 12).item())
            bias, weight = scale * bias, scale * weight
            ring.set_delay_input(scale * later)
        ring.set_input(bias, weight)
    if cfg.get("loop_power", 0.0):
        ring.set_feedback(math.sqrt(cfg["loop_power"]) * complex(math.cos(cfg["loop_phase"]),
                                                                 math.sin(cfg["loop_phase"])))
    return ring


def cache_path(task, cfg, n_modes, steps, cache, seed, noise):
    suffix = "" if (seed, noise) == (0, "none") else f"__s{seed}_{noise}"
    if N_TOTAL != N_SYMBOLS:
        suffix += f"__n{N_TOTAL}"
    return os.path.join(cache, f"{task}__{cfg['group']}__{cfg['trial']:03d}__{n_modes}x{steps}{suffix}.npz")


def with_noise(ring, noise):
    """Switch on the optical noise sources of a cache tag (``ring`` may be a stack of rings)."""
    sources = OPTICAL_NOISE[noise]
    return ring.set_noise(rin=REALISTIC["rin"] if "rin" in sources else 0.0,
                          detuning=REALISTIC["detuning"] if "detuning" in sources else 0.0,
                          detuning_corr=REALISTIC.get("detuning_corr", 1.0),
                          field=REALISTIC["field"] * ring.quantum_field_noise() if "field" in sources else 0.0)


def initial_states(ring, cfg):
    """Starting fields of the two runs of one operating point."""
    start = ring.soliton_state if cfg["group"].endswith("_soliton") else ring.init_state
    return torch.cat([start(1, seed=s) for s in (0, 1)])


def store(path, lines, tail, q_total):
    """Write one operating point to the cache.

    ``lines``: the central comb lines of its two runs, [2, symbols, CACHED_LINES];
    ``tail``: the full spectra of the last 200 symbols, from which the number of comb lines
    (within 60 dB of the strongest) is counted. A run that diverged stays non-finite to the end.
    """
    last = tail[0].mean(0)
    comb_lines = int((last > last.max() * 1e-6).sum()) if torch.isfinite(tail).all() else -1
    lines = lines.cpu()
    np.savez_compressed(path, lines=lines.numpy().astype(np.float32), q_total=q_total, comb_lines=comb_lines)
    return lines.float().double(), q_total, comb_lines     # same rounding as a cache hit


def spectra_of(task, cfg, n_modes, steps, cache, seed, noise):
    """Central comb-line powers [2 runs, symbols, CACHED_LINES] and device info, simulated or from cache."""
    path = cache_path(task, cfg, n_modes, steps, cache, seed, noise)
    if os.path.exists(path):
        blob = np.load(path)
        return torch.tensor(blob["lines"], dtype=torch.float64), float(blob["q_total"]), int(blob["comb_lines"])
    u, _ = make_task(task, WARMUP + N_TOTAL, seed=seed)
    u = torch.tensor(u, dtype=torch.float64)
    ring = with_noise(build(cfg, n_modes, steps), noise)
    spectra = simulate(ring, u, n=N_TOTAL, state=initial_states(ring, cfg),
                       u2=delayed(u, cfg) if cfg.get("delay", 0) else None)
    return store(path, select_lines(spectra, CACHED_LINES), spectra[:, -200:], ring.q_total.item())


def family(cfg):
    """What operating points must share to be simulated as one stack: input lines, symbol time, material."""
    return (len(input_lines(cfg["group"])), cfg.get("symbol_time", 1e-9), cfg.get("n_group", SILICA["n_group"]),
            cfg.get("gamma", SILICA["gamma"]))


def delayed_rows(cfgs, u, repeats):
    """Delayed signals of a stack of operating points, ``repeats`` rows each ([rows, T]), or None if none of
    them has a delay line."""
    if not any(cfg.get("delay", 0) for cfg in cfgs):
        return None
    return torch.stack([delayed(u, cfg) for cfg in cfgs]).repeat_interleave(repeats, 0)


def prefill(jobs, device, chunk=256, block=200):
    """Simulate the jobs whose spectra are not cached yet on ``device``, many operating points per batch.

    ``jobs`` are those of ``run``. Operating points that share the task, the series, the grid, the
    noise setting and the set of input lines (none, or the comb lines of the group) go through the solver
    together as one stack of rings, ``chunk`` at a time, which is what makes a GPU worthwhile.
    Afterwards ``run`` finds everything in the cache.

    Without noise the result equals that of ``spectra_of`` up to rounding. With noise it is a
    different, equally valid realisation: the noise is drawn on the device, and the series is fed
    in blocks of ``block`` symbols (to keep only the lines that are cached), at the start of which
    the detuning jitter is drawn afresh.
    """
    todo = {}
    for task, cfg, n_modes, steps, cache, _, seed, noise in jobs:
        path = cache_path(task, cfg, n_modes, steps, cache, seed, noise)
        if not os.path.exists(path):
            key = (task, seed, n_modes, steps, noise, *family(cfg))
            todo.setdefault(key, {})[path] = cfg
    for batch_index, ((task, seed, n_modes, steps, noise, *_), group) in enumerate(todo.items()):
        u = torch.tensor(make_task(task, WARMUP + N_TOTAL, seed=seed)[0], dtype=torch.float64, device=device)
        items = list(group.items())
        for start in range(0, len(items), chunk):
            part = items[start: start + chunk]
            rings = [build(cfg, n_modes, steps) for _, cfg in part]
            state = torch.cat([initial_states(ring, cfg) for ring, (_, cfg) in zip(rings, part)]).to(device)
            stack = with_noise(stack_rings(rings, repeats=2).to(device), noise)
            u2 = delayed_rows([cfg for _, cfg in part], u, 2)
            generator = torch.Generator(device=device).manual_seed(7919 * seed + 1 + 104729 * (batch_index + start))
            kept, tail, loop = [], None, None
            with torch.no_grad():
                for t0 in range(0, len(u), block):
                    spectra, state = stack(u[t0: t0 + block].expand(2 * len(part), -1), state=state,
                                           generator=generator, loop=loop,
                                           u2=None if u2 is None else u2[:, t0: t0 + block])
                    loop = stack.last_loop
                    if t0 + block > WARMUP:
                        kept.append(select_lines(spectra[:, max(WARMUP - t0, 0):], CACHED_LINES).float().cpu())
                    tail = spectra if tail is None else torch.cat([tail, spectra], 1)[:, -200:]
            lines = torch.cat(kept, 1)
            for i, (ring, (path, _)) in enumerate(zip(rings, part)):
                store(path, lines[2 * i: 2 * i + 2], tail[2 * i: 2 * i + 2], ring.q_total.item())
            del stack, tail, lines


def run(job):
    task, cfg, n_modes, steps, cache, variants, seed, noise = job
    torch.set_num_threads(1)
    lines, q_total, comb_lines = spectra_of(task, cfg, n_modes, steps, cache, seed, noise)
    _, y = make_task(task, WARMUP + N_TOTAL, seed=seed)
    y = torch.tensor(y, dtype=torch.float64)[WARMUP:]
    detector, shot_time = detection(cfg, noise)
    row = {"task": task, "series_seed": seed, **cfg, "n_modes": n_modes, "q_total": q_total,
           "comb_lines": comb_lines}
    for variant in variants:
        n_lines, taps = variant
        try:
            if comb_lines < 0:
                raise FloatingPointError
            s = score_spectra(lines, y, n_lines, taps=taps, detector_noise=detector, shot_time=shot_time,
                              error_rate=symbol_error_rate if is_symbol_task(task) else None, n_test=N_TEST)
            scores = (s["val_nmse_cross"], s["nmse_cross"], s["consistency"], s.get("error_rate_cross", float("nan")))
        except (FloatingPointError, RuntimeError):      # diverged simulation or singular fit
            scores = (float("inf"), float("inf"), 0.0, float("nan"))
        row.update(dict(zip((f"{p}_{tag(variant)}" for p in ("val", "test", "consistency", "ser")), scores)))
    return row


def best_of(rows, variant):
    """The trial with the lowest validation error."""
    return min(rows, key=lambda r: r[f"val_{tag(variant)}"])


def gmean(values):
    return float(np.exp(np.mean(np.log(np.asarray(values, dtype=float)))))


def baselines(task, variants, seeds, noise):
    """Baseline test NMSE, geometric mean over the series. With noise, baselines see noisy inputs."""
    input_noise = REALISTIC["baseline_input_noise"] if noise == "realistic" else 0.0
    per_seed = []
    for seed in seeds:
        u, y = (torch.tensor(a, dtype=torch.float64) for a in make_task(task, WARMUP + N_TOTAL, seed=seed))
        out = {}
        for n_lines, taps in variants:
            b = score_baselines(u, n=N_TOTAL, y=y, esn_sizes=(n_lines,), taps=taps, input_noise=input_noise,
                                n_test=N_TEST)
            out[f"esn_{tag((n_lines, taps))}"] = b[f"esn{n_lines}"]
            out.update({k: v for k, v in b.items() if k.startswith("linear")})
        out["linear_best"] = min(v for k, v in out.items() if k.startswith("linear_ar"))
        per_seed.append(out)
    return {k: gmean([o[k] for o in per_seed]) for k in per_seed[0]}


def fresh_lines(task, cfgs, n_modes, steps, seed, noise, pairs, device, block=200, salt=0, max_rows=480,
                shared_rin=False):
    """New runs of chosen operating points, for an error estimate that the choice has not touched.

    The point with the lowest validation error among many is, on the runs it was chosen on, better
    than it really is: validation and test share the fitted readout and the noise realisation. This
    returns, per operating point, the central comb lines of ``2 * pairs`` runs on series ``seed``
    ([runs, symbols, CACHED_LINES]) whose noise is independent of the cached runs. Points with the
    same set of input lines are simulated together as one stack, at most ``max_rows`` runs at a time.
    Different ``salt`` values give independent sets of runs (used to choose, and then to report).
    The lines are returned in single precision, as in the cache.
    ``shared_rin``: run r of every point gets the same source intensity noise (several rings fed from one
    laser); the other noise sources stay independent per point.
    """
    u = torch.tensor(make_task(task, WARMUP + N_TOTAL, seed=seed)[0], dtype=torch.float64, device=device)
    rows, out, families = 2 * pairs, [None] * len(cfgs), {}
    rin_runs = None
    if shared_rin and "rin" in OPTICAL_NOISE[noise]:
        gen = torch.Generator().manual_seed(3_000_017 + 7919 * seed + 1_000_003 * salt)
        rin_runs = torch.randn(rows, len(u), generator=gen, dtype=torch.float64).to(device)
    for i, cfg in enumerate(cfgs):
        families.setdefault(family(cfg), []).append(i)
    per_batch = max(1, max_rows // rows)
    batches = [(family, members[k: k + per_batch]) for family, members in sorted(families.items())
               for k in range(0, len(members), per_batch)]
    for index, ((lines_in, *_), members) in enumerate(batches):
        rings = [build(cfgs[i], n_modes, steps) for i in members]
        starts = [ring.soliton_state if cfgs[i]["group"].endswith("_soliton") else ring.init_state
                  for ring, i in zip(rings, members)]
        state = torch.cat([start(1, seed=100 + 1000 * salt + s) for start in starts for s in range(rows)]).to(device)
        stack = with_noise(stack_rings(rings, repeats=rows).to(device), noise)
        u2 = delayed_rows([cfgs[i] for i in members], u, rows)
        generator = torch.Generator(device=device).manual_seed(
            2_000_003 + 7919 * seed + lines_in + 1_000_003 * salt + 104_729 * index)
        kept, loop = [], None
        with torch.no_grad():
            for t0 in range(0, len(u), block):
                spectra, state = stack(u[t0: t0 + block].expand(len(members) * rows, -1), state=state,
                                       generator=generator, loop=loop,
                                       u2=None if u2 is None else u2[:, t0: t0 + block],
                                       rin=None if rin_runs is None
                                       else rin_runs[:, t0: t0 + block].repeat(len(members), 1))
                loop = stack.last_loop
                if t0 + block > WARMUP:
                    kept.append(select_lines(spectra[:, max(WARMUP - t0, 0):], CACHED_LINES).float().cpu())
        lines = torch.cat(kept, 1)
        for j, i in enumerate(members):
            out[i] = lines[j * rows: (j + 1) * rows].clone()
        del stack, lines, kept, state
    return out


def fresh_scores(lines, y, variant, noise, seed=1000, error_rate=None, cfg=None):
    """(validation NMSE, test NMSE, test error rate) of each pair of fresh runs: readout fitted on one run,
    scored on the other. The error rate is nan without ``error_rate``. ``cfg``: the operating point, for the
    detection noise of its symbol rate (default: 1 ns symbols)."""
    detector, shot_time = detection(cfg or {}, noise)
    if not torch.isfinite(lines).all():
        return [(float("inf"), float("inf"), float("nan"))] * (len(lines) // 2)
    out = []
    for k in range(len(lines) // 2):
        try:
            s = score_spectra(lines[2 * k: 2 * k + 2].double(), y, variant[0], taps=variant[1],
                              detector_noise=detector, shot_time=shot_time, seed=seed + k, error_rate=error_rate,
                              n_test=N_TEST)
            out.append((s["val_nmse_cross"], s["nmse_cross"], s.get("error_rate_cross", float("nan"))))
        except (FloatingPointError, RuntimeError):
            out.append((float("inf"), float("inf"), float("nan")))
    return out


def fresh_tests(lines, y, variant, noise, error_rate=None, cfg=None):
    """(test NMSE, test error rate) of each pair of fresh runs."""
    return [(t, e) for _, t, e in fresh_scores(lines, y, variant, noise, error_rate=error_rate, cfg=cfg)]


def select_robust(task, candidates, n_modes, steps, seed, noise, pairs, device, y, variant):
    """Re-score candidate operating points on ``pairs`` pairs of new runs and return, per group, the
    candidate with the lowest mean validation NMSE, with that mean. One run pair is a noisy estimate
    and the best of many such estimates is often a point that was merely lucky; the mean over new
    runs is not. Test errors of these runs are not used."""
    flat = [c for group in candidates for c in candidates[group]]
    lines = fresh_lines(task, [{k: c[k] for k in CFG_KEYS} for c in flat], n_modes, steps, seed, noise, pairs,
                        device, salt=1)
    means = [float(np.mean([v for v, *_ in fresh_scores(mine, y, variant, noise, seed=5000, cfg=c)]))
             for mine, c in zip(lines, flat)]
    out, i = {}, 0
    for group in candidates:
        k = len(candidates[group])
        j = int(np.argmin(means[i: i + k]))
        out[group] = (candidates[group][j], means[i + j])
        i += k
    return out


def esn_fresh(u, y, size, taps, input_noise, pairs, select_pairs=1, error_rate=None):
    """The ESN baseline under the protocol of the fresh runs: hyper-parameters chosen on the validation
    part, averaged over ``select_pairs`` realisations of the input noise (1: as in ``score_baselines``),
    test error re-measured with new realisations. Returns the mean test NMSE and the mean test error
    rate (nan without ``error_rate``)."""
    n_train, n_val = _splits(N_TOTAL, N_TEST)
    target = y[WARMUP: WARMUP + N_TOTAL]

    def errors(radius, scale, noise_seed):
        gen = torch.Generator().manual_seed(noise_seed)
        seen = u + input_noise * torch.randn(u.shape, generator=gen, dtype=u.dtype) if input_noise else u
        states = add_taps(esn_states(seen[: WARMUP + N_TOTAL], size, radius, scale), taps)[WARMUP:]
        pred = fit_readout(states, target, n_train, n_val).predict(states)
        test = slice(n_train + n_val, None)
        return (nmse(target[n_train:n_train + n_val], pred[n_train:n_train + n_val]).item(),
                nmse(target[test], pred[test]).item(),
                error_rate(target[test], pred[test]) if error_rate else float("nan"))

    _, radius, scale = min((np.mean([errors(r, s, k)[0] for k in range(select_pairs)]), r, s)
                           for r in (0.5, 0.8, 0.95, 1.1) for s in (0.1, 0.3, 1.0))
    scores = [errors(radius, scale, 100 + k) for k in range(pairs)]
    return float(np.mean([s[1] for s in scores])), float(np.mean([s[2] for s in scores]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=32, help="random operating points per encoding")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--groups", nargs="+", default=list(GROUPS), choices=ALL_GROUPS,
                    help="encodings to compare; the mask ablations (comb_mask4 ... comb_amplitude) are opt-in")
    ap.add_argument("--series-seeds", type=int, nargs="+", default=[0],
                    help="independent realisations of each task; results are aggregated over them")
    ap.add_argument("--noise", default="none", choices=["none", "realistic"],
                    help="realistic = laser intensity noise 0.1 percent, detuning jitter 1 percent of the "
                         "half-linewidth, quantum-level field noise, photon shot noise and 0.3 uW "
                         "detector noise per comb line")
    ap.add_argument("--n-modes", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="encodings")
    ap.add_argument("--variants", nargs="+", default=[tag(v) for v in VARIANTS],
                    help="readouts to score, as <comb lines>x<taps> (at most %d lines)" % CACHED_LINES)
    ap.add_argument("--main", default=tag(MAIN), help="variant used for the fine-grid re-check and the plot")
    ap.add_argument("--cache", help="folder of cached spectra (default: <out>/<name>_spectra)")
    ap.add_argument("--no-fine-grid", action="store_true", help="skip the finer-grid re-check")
    ap.add_argument("--chunk", type=int, default=256, help="operating points per GPU batch")
    ap.add_argument("--fresh-pairs", type=int, default=0,
                    help="re-measure each chosen operating point (and the ESN) on this many pairs of new runs with "
                         "independent noise; removes the optimism of reporting a point on the runs it was chosen on")
    ap.add_argument("--select-top", type=int, default=0,
                    help="with --fresh-pairs: choose each point among the best this many (by validation error on "
                         "the cached runs) by their mean validation error on --select-pairs pairs of new runs")
    ap.add_argument("--select-pairs", type=int, default=3)
    ap.add_argument("--test-symbols", type=int, default=0,
                    help="test part of this many symbols after the unchanged training and validation parts "
                         "(%d + %d symbols); default: the last 20 %% of %d symbols" % (
                             int(0.6 * N_SYMBOLS), int(0.2 * N_SYMBOLS), N_SYMBOLS))
    ap.add_argument("--detector", type=float, default=None,
                    help="detector noise in uW rms per comb line for --noise realistic (default %g); 0 = ideal "
                         "detectors, which leaves the optical noise and photon shot noise: the noise floor" % (
                             REALISTIC["detector"] * 1e6))
    ap.add_argument("--device", default="cpu",
                    help="'cuda' simulates on the GPU, many operating points per batch; scoring stays on the CPU")
    args = ap.parse_args()
    variants = tuple(tuple(int(x) for x in v.split("x")) for v in args.variants)
    main_variant = tuple(int(x) for x in args.main.split("x"))
    if main_variant not in variants:
        ap.error("--main must be one of --variants")
    seeds, main_tag = args.series_seeds, tag(main_variant)
    set_length(args.test_symbols, args.detector)

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, args.name)
    cache = args.cache or stem + "_spectra"
    os.makedirs(cache, exist_ok=True)

    jobs = [(task, sample(group, trial), args.n_modes, args.steps, cache, variants, seed, args.noise)
            for seed in seeds for task in args.tasks for group in args.groups for trial in range(args.trials)]
    if args.device != "cpu":
        prefill(jobs, args.device, chunk=args.chunk)
    with Pool(args.workers, initializer=set_length, initargs=(args.test_symbols, args.detector)) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    with open(stem + "_trials.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def trials(task, group, seed):
        return [r for r in rows if r["task"] == task and r["group"] == group and r["series_seed"] == seed]

    # Re-simulate, on a finer grid, the operating point chosen under the main readout on the first series.
    fine = {}
    if not args.no_fine_grid:
        picks = {(t, g): best_of(trials(t, g, seeds[0]), main_variant) for t in args.tasks for g in args.groups}
        fine_jobs = [(t, {k: c[k] for k in CFG_KEYS}, 2 * args.n_modes, 2 * args.steps, cache, variants,
                      seeds[0], args.noise) for (t, g), c in picks.items()]
        if args.device != "cpu":
            prefill(fine_jobs, args.device)
        with Pool(args.workers, initializer=set_length, initargs=(args.test_symbols, args.detector)) as pool:
            fine = dict(zip(picks, pool.map(run, fine_jobs, chunksize=1)))

    # Error of the chosen points on runs with new noise: the choice cannot have profited from these.
    fresh, esn_new, esn_ser, robust = {}, {}, {}, {}
    if args.fresh_pairs:
        input_noise = REALISTIC["baseline_input_noise"] if args.noise == "realistic" else 0.0
        key = f"val_{main_tag}"
        for task in args.tasks:
            per_seed, per_seed_ser = [], []
            error_rate = symbol_error_rate if is_symbol_task(task) else None
            for seed in seeds:
                u, y = (torch.tensor(a, dtype=torch.float64) for a in make_task(task, WARMUP + N_TOTAL, seed=seed))
                if args.select_top:
                    candidates = {g: sorted(trials(task, g, seed), key=lambda r: r[key])[: args.select_top]
                                  for g in args.groups}
                    chosen = select_robust(task, candidates, args.n_modes, args.steps, seed, args.noise,
                                           args.select_pairs, args.device, y[WARMUP:], main_variant)
                    for g, (row, val) in chosen.items():
                        robust[(task, g, seed)] = (row, val, candidates[g][0]["trial"])
                    picks = [chosen[g][0] for g in args.groups]
                else:
                    picks = [best_of(trials(task, g, seed), main_variant) for g in args.groups]
                lines = fresh_lines(task, [{k: c[k] for k in CFG_KEYS} for c in picks], args.n_modes, args.steps,
                                    seed, args.noise, args.fresh_pairs, args.device)
                for group, mine, c in zip(args.groups, lines, picks):
                    fresh[(task, group, seed)] = fresh_tests(mine, y[WARMUP:], main_variant, args.noise,
                                                             error_rate, cfg=c)
                test, ser = esn_fresh(u, y, main_variant[0], main_variant[1], input_noise, args.fresh_pairs,
                                      args.select_pairs if args.select_top else 1, error_rate)
                per_seed.append(test)
                per_seed_ser.append(ser)
            esn_new[task] = gmean(per_seed)
            esn_ser[task] = float(np.mean(per_seed_ser))

    summary = []
    for task in args.tasks:
        base = baselines(task, variants, seeds, args.noise)
        print(f"\n=== {task}   cross-run test NMSE, geometric mean over {len(seeds)} series, noise: {args.noise}"
              f"   (columns: comb lines x readout taps)"
              + (f"   test part {N_TEST} symbols" if N_TEST else ""))
        print(f"{'':18s}" + "".join(f"{tag(v):>10s}" for v in variants)
              + f"{'min..max ' + main_tag:>23s}{'fine ' + main_tag:>11s}{'consist.':>9s}{'SER':>8s}"
              + f"   point chosen on series {seeds[0]} for {main_tag}")
        print(f"{'linear AR (best)':18s}" + f"{base['linear_best']:10.2e}" * len(variants))
        print(f"{'ESN, same size':18s}" + "".join(f"{base['esn_' + tag(v)]:10.2e}" for v in variants))
        if fresh:
            print(f"with new noise after the choice ({main_tag}, mean of {args.fresh_pairs} pairs of runs per series):"
                  f"   ESN {esn_new[task]:.2e}"
                  + (f", SER {esn_ser[task]:.2e}" if is_symbol_task(task) else ""))
        for group in args.groups:
            chosen = {v: [best_of(trials(task, group, s), v) for s in seeds] for v in variants}
            main_rows = chosen[main_variant]
            tests = [r[f"test_{main_tag}"] for r in main_rows]
            c = robust[(task, group, seeds[0])][0] if robust else main_rows[0]
            entry = {"task": task, "group": group, "noise": args.noise, "n_series": len(seeds),
                     "min_test": min(tests), "max_test": max(tests),
                     "fine_grid_test": fine[(task, group)][f"test_{main_tag}"] if fine else float("nan"),
                     "first_series_test": tests[0],
                     "consistency": float(np.mean([r[f"consistency_{main_tag}"] for r in main_rows])),
                     "ser": float(np.mean([r[f"ser_{main_tag}"] for r in main_rows])),
                     "trial": c["trial"], "q_total": c["q_total"], "p_total": c["p_total"], "zeta": c["zeta"],
                     "depth": c["depth"], "baseline_linear": base["linear_best"]}
            for v in variants:
                entry[f"best_test_{tag(v)}"] = gmean([r[f"test_{tag(v)}"] for r in chosen[v]])
                entry[f"baseline_esn_{tag(v)}"] = base["esn_" + tag(v)]
            note = ""
            if fresh:
                mine = [np.asarray([t for t, _ in fresh[(task, group, s)]]) for s in seeds]
                entry["fresh_test"] = gmean([m.mean() for m in mine])
                entry["fresh_spread"] = float(np.mean([m.std(ddof=1) / m.mean() for m in mine]))
                entry["baseline_esn_fresh"] = esn_new[task]
                note = f"   new noise {entry['fresh_test']:.2e} (pairs differ by {entry['fresh_spread']:.0%})"
                if is_symbol_task(task):
                    entry["fresh_ser"] = float(np.mean([e for s in seeds for _, e in fresh[(task, group, s)]]))
                    entry["baseline_esn_fresh_ser"] = esn_ser[task]
                    note += f", SER {entry['fresh_ser']:.2e}"
            if robust:
                mine = [robust[(task, group, s)] for s in seeds]
                entry["robust_val"] = gmean([m[1] for m in mine])
                entry["robust_changed"] = sum(m[0]["trial"] != m[2] for m in mine)
                entry["robust_trial"] = c["trial"]
                note += f"; chosen on {args.select_pairs} new pairs, val {entry['robust_val']:.2e}, " \
                        f"other point than single-run choice in {entry['robust_changed']}/{len(seeds)} series"
            summary.append(entry)
            span = f"{entry['min_test']:.1e}..{entry['max_test']:.1e}"
            print(f"{group:18s}" + "".join(f"{entry['best_test_' + tag(v)]:10.2e}" for v in variants)
                  + f"{span:>23s}{entry['fine_grid_test']:11.2e}{entry['consistency']:9.3f}{entry['ser']:8.4f}"
                  + f"   Q {c['q_total']:.1e}, {c['p_total'] * 1e3:.0f} mW, zeta {c['zeta']:.1f}, depth {c['depth']:.2f}" + note)
    with open(stem + "_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plain = (main_variant[0], 1) if (main_variant[0], 1) in variants else variants[0]
    fig, axes = plt.subplots(1, len(args.tasks), figsize=(5 * len(args.tasks), 4.4), squeeze=False)
    for ax, task in zip(axes[0], args.tasks):
        mine = [s for s in summary if s["task"] == task]
        x = np.arange(len(mine))
        ax.bar(x - 0.2, [s[f"best_test_{tag(plain)}"] for s in mine], 0.4, color="C7",
               label=f"readout sees the current symbol only ({tag(plain)})")
        best = np.array([s[f"best_test_{main_tag}"] for s in mine])
        err = np.array([[max(b - s["min_test"], 0), max(s["max_test"] - b, 0)] for b, s in zip(best, mine)]).T
        colors = ["C2" if s["group"].startswith("comb") else "C1" if s["group"].endswith("soliton") else "C0" for s in mine]
        ax.bar(x + 0.2, best, 0.4, color=colors, yerr=err, capsize=2,
               label=f"readout with {main_variant[1]} taps ({main_tag}); bars: min..max over series")
        if fine:
            ax.plot(x + 0.2, [s["fine_grid_test"] for s in mine], "kx", label="first series, finer grid")
        ax.axhline(mine[0]["baseline_linear"], color="gray", linestyle="--", label="best linear AR")
        ax.axhline(mine[0][f"baseline_esn_{main_tag}"], color="black", linestyle=":", label=f"ESN ({main_tag})")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([s["group"].replace("_", "\n") for s in mine], fontsize=7)
        ax.set_title(f"{task} (noise: {args.noise})")
    axes[0][0].set_ylabel(f"cross-run test NMSE, best of N, mean of {len(seeds)} series")
    axes[0][0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(stem + ".png", dpi=130)
    print("\nsaved", stem + "_trials.csv, _summary.csv and .png; spectra cached in", cache)


if __name__ == "__main__":
    main()
