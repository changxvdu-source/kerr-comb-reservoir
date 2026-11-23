"""Can a reservoir forecast volatility? A walk-forward test on a public daily index series.

Returns themselves are close to unpredictable (finance_walkforward.py); their size is not: large moves
cluster. This compares volatility forecasters on the daily Nasdaq Composite (FRED NASDAQCOM, log returns
r[t] in units of the standard deviation of the first ``--initial`` days).

Target, for horizon h: the mean squared return of the next h days, RV_h[t] = mean(r[t+1]^2 ... r[t+h]^2),
a noisy but unbiased proxy of the variance (only closing prices are public here).

Forecasters (all fitted on past days only, refitted every ``--refit`` days, expanding window):
  mean        the mean of r^2 over the training window
  GARCH       GARCH(1,1) by Gaussian maximum likelihood, h-day mean of its variance forecast
  HAR         log-HAR: ridge regression of log RV_h on log of the 1-, 5- and 22-day mean squared returns
  LHAR        HAR plus the leverage terms r[t] and min(r[t], 0) (falls raise volatility more than rises)
  ESN n       leaky tanh echo state network driven by r[t] and log r[t]^2, ridge readout on its states
              and the inputs; reservoir (spectral radius, input scale, leak rate, seed) and ridge strength
              chosen on the last 20 % of each training window
  LHAR+ESN    the LHAR inputs and the ESN 100 states together (does the reservoir add anything?)
The regressions forecast log RV_h; back on the variance scale with the mean of exp(residual) on the
training rows (smearing), so the forecasts are unbiased on average.

Scores per horizon, over all forecast days and from 2000 on:
  R2          1 - sum (RV - f)^2 / sum (RV - f_mean)^2     (squared error on the variance scale)
  QLIKE       mean(RV / f + log f) minus the same for "mean" (< 0: better than the mean; robust to the
              noise of the proxy, Patton 2011)
  DM t        Diebold-Mariano statistic of the QLIKE difference against HAR (> 2: better than HAR),
              Newey-West standard error with h + 5 lags

    python experiments/finance_volatility.py data/fred_nasdaqcom/NASDAQCOM.csv --horizons 1 5 22
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.signal import lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from combrc.readout import Ridge  # noqa: E402
from finance_walkforward import load  # noqa: E402

LAMBDAS = (1e-3, 1e-1, 10.0, 1e3)
ESN_GRID = [(rad, s, leak, e) for rad in (0.5, 0.9) for s in (0.1, 0.5) for leak in (1.0, 0.2) for e in range(2)]
WARMUP = 100
EPS = 1e-4          # added to squared returns before the logarithm (z units; a 1 % move is about 1.3 z)


def trailing_mean(x, w):
    """Mean of x[t-w+1 .. t] (fewer at the start)."""
    c = np.cumsum(np.concatenate([[0.0], x]))
    t = np.arange(1, len(x) + 1)
    return (c[t] - c[np.maximum(t - w, 0)]) / np.minimum(t, w)


def forward_mean(x, h):
    """Mean of x[t+1 .. t+h]; nan where the window runs past the end."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(x), np.nan)
    t = np.arange(len(x) - h)
    out[t] = (c[t + h + 1] - c[t + 1]) / h
    return out


def esn_states(inputs, n, radius, scale, leak, seed):
    """Leaky tanh ESN driven by ``inputs`` [T, k] -> states [T, n]."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, n))
    w *= radius / np.abs(np.linalg.eigvals(w)).max()
    w_in = scale * rng.uniform(-1, 1, (n, inputs.shape[1]))
    bias = 0.2 * rng.uniform(-1, 1, n)
    x, out = np.zeros(n), np.empty((len(inputs), n))
    drive = inputs @ w_in.T + bias
    for t in range(len(inputs)):
        x = (1 - leak) * x + leak * np.tanh(w @ x + drive[t])
        out[t] = x
    return out


# ---- GARCH(1,1) -----------------------------------------------------------------------------------

def garch_variance(z2, omega, alpha, beta, v0):
    """sigma^2[t] (variance of z[t] given z[..t-1]) for t = 0 .. T, i.e. one more than len(z2)."""
    e = np.concatenate([[v0], omega + alpha * z2])
    return lfilter([1.0], [1.0, -beta], e)


def garch_params(theta):
    p = 1 / (1 + math.exp(-theta[1]))            # persistence alpha + beta, in (0, 1)
    q = 1 / (1 + math.exp(-theta[2]))            # share of alpha
    return math.exp(theta[0]), p * q, p * (1 - q)


def fit_garch(z2):
    v0 = z2.mean()

    def nll(theta):
        omega, alpha, beta = garch_params(theta)
        s = garch_variance(z2, omega, alpha, beta, v0)[:-1]
        return 0.5 * np.sum(np.log(s) + z2 / s)
    res = minimize(nll, x0=[math.log(0.05 * v0), 3.0, -2.0], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000})
    return garch_params(res.x)


# ---- regression forecasters ------------------------------------------------------------------------

def choose_and_fit(sets, y, train_idx):
    """(feature set, ridge strength) chosen on the last 20 % of the training rows by squared error of log RV,
    then refitted on all of them; returns (name, model, smearing factor)."""
    n = len(train_idx)
    fit_rows, val_rows = train_idx[: int(0.8 * n)], train_idx[int(0.8 * n):]
    best = (float("inf"), None, None)
    for name, x in sets.items():
        xt, xv = torch.tensor(x[fit_rows]), torch.tensor(x[val_rows])
        yt, yv = torch.tensor(y[fit_rows]), torch.tensor(y[val_rows])
        for lam in LAMBDAS:
            model = Ridge(lam, eps=1e-6).fit(xt, yt)
            err = float(((model.predict(xv) - yv) ** 2).mean())
            if err < best[0]:
                best = (err, name, lam)
    _, name, lam = best
    x_all, y_all = torch.tensor(sets[name][train_idx]), torch.tensor(y[train_idx])
    model = Ridge(lam, eps=1e-6).fit(x_all, y_all)
    smear = float(torch.exp(y_all - model.predict(x_all)).mean())
    return name, model, smear


def dm_stat(loss_ref, loss_model, lag):
    d = loss_ref - loss_model
    n, mean = len(d), d.mean()
    c = d - mean
    var = c @ c / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (c[k:] @ c[:-k]) / n
    return mean / math.sqrt(var / n) if var > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 22])
    ap.add_argument("--initial", type=int, default=2520)
    ap.add_argument("--refit", type=int, default=252)
    ap.add_argument("--sizes", type=int, nargs="+", default=[31, 100])
    ap.add_argument("--clip", type=float, default=8.0,
                    help="the ESN's signed input r[t] is clipped to +-clip standard deviations (its log r^2 input is not)")
    ap.add_argument("--out", default="results/finance_volatility")
    args = ap.parse_args()
    torch.set_num_threads(2)

    dates, r = load(args.csv)
    z = r / r[: args.initial].std()
    z2 = z ** 2
    T = len(z)
    print(f"{T} returns {dates[0]} .. {dates[-1]}", flush=True)

    # features of row t use z[0..t] only
    har = np.stack([np.log(trailing_mean(z2, w) + EPS) for w in (1, 5, 22)], 1)
    lhar = np.hstack([har, z[:, None], np.minimum(z, 0)[:, None]])
    inputs = np.stack([z, (np.log(z2 + EPS) + 1.27) / 2.2], 1)       # log z^2 of a unit Gaussian: mean -1.27, sd 2.2
    inputs[:, 0] = np.clip(inputs[:, 0], -args.clip, args.clip)
    feats = {"HAR": {"har": har}, "LHAR": {"lhar": lhar}}
    esn100 = {}
    for n in args.sizes:
        sets = {}
        for rad, s, leak, e in ESN_GRID:
            states = esn_states(inputs, n, rad, s, leak, e)
            sets[f"r={rad} s={s} leak={leak} seed={e}"] = np.hstack([states, inputs])
            if n == max(args.sizes):
                esn100[f"r={rad} s={s} leak={leak} seed={e}"] = np.hstack([states, lhar])
        feats[f"ESN {n}"] = sets
        print(f"ESN {n}: {len(sets)} reservoirs", flush=True)
    feats["LHAR+ESN"] = esn100

    table, first = [], max(args.initial, WARMUP + 1)
    garch_fits = {}                                       # GARCH does not depend on the horizon: fit once per refit
    saved = {}                                            # forecasts for later inspection
    for h in args.horizons:
        target = forward_mean(z2, h)                      # RV_h[t], known only after day t + h
        y = np.log(target + EPS)
        models = ["mean", "GARCH", *feats]
        f = {k: np.full(T, np.nan) for k in models}
        chosen = {k: [] for k in feats}
        for start in range(first, T - h, args.refit):
            train_idx = np.arange(WARMUP, start - h)        # target window ends before the forecast day
            test_idx = np.arange(start, min(start + args.refit, T - h))
            past = z2[: start]                               # squared returns known on the forecast day ...
            f["mean"][test_idx] = past[WARMUP:].mean()
            if start not in garch_fits:
                garch_fits[start] = fit_garch(past[WARMUP:])
            omega, alpha, beta = garch_fits[start]
            s = garch_variance(z2, omega, alpha, beta, past[WARMUP:].mean())   # s[t+1]: forecast made on day t
            p, longrun = alpha + beta, omega / (1 - alpha - beta)
            nxt = s[test_idx + 1]
            f["GARCH"][test_idx] = longrun + (nxt - longrun) * (1 - p ** h) / ((1 - p) * h)
            for k, sets in feats.items():
                name, model, smear = choose_and_fit(sets, y, train_idx)
                f[k][test_idx] = smear * np.exp(model.predict(torch.tensor(sets[name][test_idx])).numpy())
                chosen[k].append(name)
        rows = np.arange(first, T - h)
        rows = rows[~np.isnan(f["mean"][rows])]
        rv = target[rows]
        saved[f"h{h}_dates"] = np.array([dates[t] for t in rows])
        saved[f"h{h}_rv"] = rv
        saved[f"h{h}_forecasts"] = np.stack([f[k][rows] for k in models])
        saved["models"] = np.array(models)
        years = np.array([int(dates[t + 1][:4]) for t in rows])
        for label, mask in (("1981-2026", np.ones(len(rows), bool)), ("2000-2026", years >= 2000)):
            ref_mse = np.sum((rv[mask] - f["mean"][rows][mask]) ** 2)
            ql = {k: rv[mask] / f[k][rows][mask] + np.log(f[k][rows][mask]) for k in models}
            for k in models:
                fk = f[k][rows][mask]
                table.append({"horizon": h, "period": label, "model": k, "days": int(mask.sum()),
                              "r2_vs_mean": 1 - np.sum((rv[mask] - fk) ** 2) / ref_mse,
                              "qlike_vs_mean": float(ql[k].mean() - ql["mean"].mean()),
                              "dm_t_vs_har": dm_stat(ql["HAR"], ql[k], h + 5) if k != "HAR" else float("nan")})
        print(f"\nhorizon {h} day(s)")
        print(f"  {'model':9s} {'period':10s} {'R2':>8s} {'QLIKE':>8s} {'DM t vs HAR':>12s}")
        for row in table:
            if row["horizon"] == h:
                print(f"  {row['model']:9s} {row['period']:10s} {row['r2_vs_mean']:8.3f} {row['qlike_vs_mean']:8.3f} "
                      f"{row['dm_t_vs_har']:12.2f}")
        for k in (f"ESN {n}" for n in args.sizes):
            counts = {}
            for name in chosen[k]:
                counts[name] = counts.get(name, 0) + 1
            print(f"  {k} chosen: " + ", ".join(f"{a} x{c}" for a, c in sorted(counts.items(), key=lambda kv: -kv[1])[:3]))
        sys.stdout.flush()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    stem = args.out + ("" if args.horizons == [1, 5, 22] else "_h" + "_".join(map(str, args.horizons)))
    stem += "" if args.clip == 8.0 else f"_clip{args.clip:g}"
    with open(stem + ".csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    np.savez_compressed(stem + "_forecasts.npz", **saved)
    print("saved", stem + ".csv and _forecasts.npz")


if __name__ == "__main__":
    main()
