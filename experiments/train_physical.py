"""Train the ring's physical parameters by backpropagating through the LLE solver.

Task: predict u[t + horizon] from the comb-line powers at symbol t.

Each iteration runs the ring on several windows of the training series, each for
several noise seeds, all in one batch. The ridge readout is fitted in closed form
on the first part of the windows (inside the graph, pooled over windows and
seeds) and a gradient step is taken on

    loss = NMSE on the rest of the window  +  c * inconsistency

where inconsistency is the across-seed variance of the standardised features:
0 when the response is set by the input alone, ~1 when it is set by the noise
(chaos). Pooling over seeds also means a readout cannot do well by fitting one
particular noise realisation.

Safeguards for the chaotic cliff: truncated BPTT, gradient clipping, and undoing
any step after which the loss jumps far above its running average (then the
learning rate is halved). The learning rate also follows a cosine decay. The
parameters kept are those with the best score on a
separate validation series; the reported numbers are from a third, test series.

Noise-aware training: with --rin / --detuning-noise / --field-noise /
--detector-noise the same noise is present in the training passes, in model
selection and in the reported test score (the noise-free test score is reported
as well).

Examples:
    python experiments/train_physical.py --name multimode
    python experiments/train_physical.py --name finetune --init-log-q-ext1 6.3 --init-p-avg 0.03 \n        --init-p-swing 0.01 --init-input 0.0224 --lr-scale 0.3 --iters 80
    python experiments/train_physical.py --name pump_only --input-modes 0
    python experiments/train_physical.py --name noise_aware --init-ring results/train_default_start_256.pt         --rin 1e-3 --detuning-noise 0.01 --field-noise 1 --detector-noise 3e-7 --lr-scale 0.3
    python experiments/train_physical.py --name old_method --input-modes 0 --windows 1 --seeds 1 --c-weight 0
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from combrc import (KerrRing, Ridge, load_ring, make_series, nmse, save_ring,  # noqa: E402
                    score_baselines, score_ring, select_lines, to_features)
from combrc.readout import add_taps, measure  # noqa: E402
from combrc.evaluate import N_SYMBOLS, WARMUP  # noqa: E402

# Box constraints on the raw (O(1)-scaled) parameters, keeping the device physical.
BOUNDS = {
    "_delta0": (-2.0, 6.0),             # 1e-3 rad
    "_beta2": (-40.0, -0.4),            # 1e-27 s^2/m, anomalous
    "_log_q_ext1": (5.5, 8.5),
    "_log_q_ext2": (5.5, 8.5),
    "_log_p_avg": (-3.0, math.log10(0.5)),   # 1 mW .. 0.5 W
    "_mod_depth": (0.02, 1.5),
    "_input_bias": (-0.4, 0.4),         # sqrt(W), per quadrature
    "_input_weight": (-0.4, 0.4),
    "_dint": (-5.0, 5.0),               # 1e-3 rad
}
LEARNING_RATES = {"_input_bias": 0.005, "_input_weight": 0.005}
DEFAULT_LR = 0.01
CLIFF = 4.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="run", help="output files are results/train_<name>.*")
    ap.add_argument("--task", default="mackey_glass", choices=["mackey_glass", "rossler", "lorenz"])
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--train", nargs="+", default=["delta0", "beta2", "log_q_ext1", "log_q_ext2",
                                                   "p_avg", "p_swing", "input_bias", "input_weight"])
    ap.add_argument("--input-modes", type=int, default=8,
                    help="extra input lines on each side of the pump (0 = pump-power encoding only)")
    ap.add_argument("--windows", type=int, default=3, help="training windows per iteration")
    ap.add_argument("--seeds", type=int, default=2, help="noise realisations per window")
    ap.add_argument("--c-weight", type=float, default=1.0, help="weight of the inconsistency penalty")
    ap.add_argument("--init-log-q-ext1", type=float, default=6.0)
    ap.add_argument("--init-log-q-ext2", type=float, default=7.35)
    ap.add_argument("--init-p-avg", type=float, default=0.15)
    ap.add_argument("--init-p-swing", type=float, default=0.05)
    ap.add_argument("--init-input", type=float, default=0.05,
                    help="std of the random initial input-line amplitudes, sqrt(W)")
    ap.add_argument("--lr-scale", type=float, default=1.0, help="multiplies every learning rate")
    ap.add_argument("--init-ring", help="start from a ring saved by an earlier run (overrides --init-* and --n-modes)")
    ap.add_argument("--rin", type=float, default=0.0, help="relative rms laser intensity noise per symbol")
    ap.add_argument("--detuning-noise", type=float, default=0.0, help="rms detuning jitter, half-linewidths")
    ap.add_argument("--field-noise", type=float, default=0.0, help="intracavity noise, multiples of the quantum level")
    ap.add_argument("--detector-noise", type=float, default=0.0, help="rms noise per comb-line reading, W")
    ap.add_argument("--shot-time", type=float, default=0.0,
                    help="integration time per reading, s; > 0 switches on photon shot noise at the detectors")
    ap.add_argument("--taps", type=int, default=1, help="the readout also sees the previous taps - 1 symbols")
    ap.add_argument("--init-trial", type=int,
                    help="start from this per-line-mask operating point of compare_encodings.py (overrides --init-*)")
    ap.add_argument("--max-power", type=float,
                    help="cap on the mean optical input power, W; the pump and the input lines are scaled back to it")
    ap.add_argument("--n-modes", type=int, default=256,
                    help="spectral grid; 128 is too narrow here: training then exploits the truncation")
    ap.add_argument("--no-fine-grid", action="store_true", help="skip the finer-grid re-check of the result")
    ap.add_argument("--n-lines", type=int, default=31)
    ap.add_argument("--steps", type=int, default=20, help="RK4IP steps per symbol")
    ap.add_argument("--window", type=int, default=400, help="symbols with gradient per iteration")
    ap.add_argument("--warmup", type=int, default=100, help="gradient-free symbols before the window")
    ap.add_argument("--tbptt", type=int, default=16, help="symbols of history gradients may flow through")
    ap.add_argument("--lam", type=float, default=1e-3, help="ridge strength used inside the training loss")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    k = args.input_modes
    ring = KerrRing(
        n_modes=args.n_modes, steps_per_symbol=args.steps, seed=args.seed,
        trainable=[t for t in args.train if k or not t.startswith("input_")],
        input_modes=[m for m in range(-k, k + 1) if m],
        q_ext1=10 ** args.init_log_q_ext1, q_ext2=10 ** args.init_log_q_ext2,
        p_avg=args.init_p_avg, p_swing=args.init_p_swing, input_init=args.init_input,
    )
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f"train_{args.name}")
    if args.init_trial is not None:
        import compare_encodings as ce
        save_ring(ce.build(ce.sample("comb_mask", args.init_trial), args.n_modes, args.steps), stem + ".pt")
        ring = load_ring(stem + ".pt", trainable=args.train)
    if args.init_ring:
        ring = load_ring(args.init_ring)
    floor = max(1e-9, args.detector_noise)
    detection = {"detector_noise": args.detector_noise, "shot_time": args.shot_time}

    def with_noise(model):
        return model.set_noise(rin=args.rin, detuning=args.detuning_noise,
                               field=args.field_noise * model.quantum_field_noise())
    f64 = torch.float64
    need = WARMUP + N_SYMBOLS + args.horizon
    series = {name: torch.tensor(make_series(args.task, n, seed=s), dtype=f64)
              for name, n, s in (("train", 6000, 100), ("val", need, 200), ("test", need, 0))}

    def score(split, model=None, noisy=True):
        model = model or ring
        model = with_noise(model) if noisy else model.set_noise()
        return score_ring(model, series[split], args.horizon, args.n_lines, taps=args.taps,
                          **(detection if noisy else {}))

    summary = {"args": vars(args), "baselines_test": score_baselines(series["test"], args.horizon),
               "before": {"test": score("test"), "params": ring.describe()}}
    print("baselines (test NMSE):", {a: f"{b:.2e}" for a, b in summary["baselines_test"].items()})
    print("before training, test:", summary["before"]["test"], flush=True)

    groups = [{"params": [p], "lr": args.lr_scale * LEARNING_RATES.get(n, DEFAULT_LR)}
              for n, p in ring.named_parameters()]
    opt = torch.optim.Adam(groups)
    initial_lrs = [g["lr"] for g in groups]

    def snapshot():
        return {a: b.clone() for a, b in ring.state_dict().items()}

    history, ema, good, cliff_scale = [], None, snapshot(), 1.0
    best_val, best_iter = score("val")["nmse_cross"], -1
    save_ring(ring, stem + ".pt")
    n_fit = int(0.6 * args.window)
    span = args.warmup + args.window
    for it in range(args.iters):
        t0 = time.time()
        offsets = torch.randint(0, len(series["train"]) - span - args.horizon, (args.windows,)).tolist()
        # Batch rows: window 0 seed 0, window 0 seed 1, ..., i.e. [windows * seeds, span].
        u = torch.stack([series["train"][o: o + span] for o in offsets]).repeat_interleave(args.seeds, 0)
        y = torch.stack([series["train"][o + args.warmup + args.horizon: o + span + args.horizon]
                         for o in offsets]).repeat_interleave(args.seeds, 0)
        with_noise(ring)                         # the quantum level follows the current linewidth
        with torch.no_grad():
            _, state = ring(u[:, : args.warmup], state=ring.init_state(len(u), seed=None))
        spectra, _ = ring(u[:, args.warmup:], state=state, tbptt=args.tbptt)
        lines = measure(select_lines(spectra, args.n_lines), **detection)
        x = add_taps(to_features(lines, floor=floor), args.taps)             # [rows, window, lines * taps]
        n_feat = x.shape[-1]
        readout = Ridge(args.lam).fit(x[:, :n_fit].reshape(-1, n_feat), y[:, :n_fit].reshape(-1))
        task_loss = nmse(y[:, n_fit:].reshape(-1), readout.predict(x[:, n_fit:].reshape(-1, n_feat)))
        if args.seeds > 1:      # variance across seeds, for the same window
            z = readout.standardise(x).reshape(args.windows, args.seeds, -1, n_feat)
            incons = z.var(dim=1).mean()
        else:
            incons = torch.zeros((), dtype=f64)
        loss = task_loss + args.c_weight * incons

        opt.zero_grad()
        loss.backward()
        finite = math.isfinite(loss.item()) and all(torch.isfinite(p.grad).all() for p in ring.parameters())
        if not finite or (ema is not None and loss.item() > CLIFF * ema):
            ring.load_state_dict(good)                       # undo the step that led here
            cliff_scale = max(0.5 * cliff_scale, 1 / 16)
            status = "undone"
        else:
            ema = loss.item() if ema is None else 0.8 * ema + 0.2 * loss.item()
            good = snapshot()
            torch.nn.utils.clip_grad_norm_(ring.parameters(), 1.0)
            decay = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * it / args.iters))
            for g, lr0 in zip(opt.param_groups, initial_lrs):
                g["lr"] = lr0 * cliff_scale * decay
            opt.step()
            with torch.no_grad():
                for name, p in ring.named_parameters():
                    p.clamp_(*BOUNDS[name])
                excess = ring.mean_input_power().item() / args.max_power if args.max_power else 1.0
                if excess > 1:
                    ring._log_p_avg -= math.log10(excess)
                    ring._input_bias /= math.sqrt(excess)
                    ring._input_weight /= math.sqrt(excess)
            status = "step"
        d = ring.describe()
        history.append({"iter": it, "loss": loss.item(), "task": task_loss.item(),
                        "inconsistency": incons.item(), "status": status, **d})
        line = (f"it {it:3d} {status:6s} loss {loss.item():.3e} (task {task_loss.item():.3e}, "
                f"incons {incons.item():.2e})  Q {d['q_total']:.2e}  d0 {d['delta0']:.2e}  "
                f"P {d['p_avg'] * 1e3:.0f} mW  swing {d['p_swing'] * 1e3:.0f} mW  ({time.time() - t0:.0f}s)")
        if (it + 1) % args.eval_every == 0:
            val = score("val")
            line += f"  | val cross-seed NMSE {val['nmse_cross']:.3e} consistency {val['consistency']:.3f}"
            history[-1]["val"] = val
            if val["nmse_cross"] < best_val:
                best_val, best_iter = val["nmse_cross"], it
                save_ring(ring, stem + ".pt")
        print(line, flush=True)

    best = load_ring(stem + ".pt")
    summary["best_iter"] = best_iter
    summary["after"] = {"test": score("test", best), "test_noise_free": score("test", best, noisy=False),
                        "params": best.describe()}
    # Numerical re-check of the trained device: finer time step and twice the spectral grid.
    if not args.no_fine_grid:
        fine = load_ring(stem + ".pt", n_modes=2 * best.n_modes, steps_per_symbol=2 * best.steps_per_symbol)
        summary["after"]["test_fine_grid"] = score("test", fine)
    summary["history"] = history
    with open(stem + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"best validation at iteration {best_iter}")
    print("after training, test:           ", summary["after"]["test"])
    print("same device, noise switched off:", summary["after"]["test_noise_free"])
    print("same device on a finer grid:    ", summary["after"].get("test_fine_grid", "skipped"))
    print("trained parameters:", {a: f"{b:.3g}" for a, b in summary["after"]["params"].items()})
    print("saved", stem + ".json", "and .pt")


if __name__ == "__main__":
    main()
