"""Can a reservoir predict tomorrow's index return? A walk-forward test on a public daily series.

Series: daily close of the Nasdaq Composite from FRED (NASDAQCOM, 1971 onwards), log returns r[t].
Task: predict r[t+1] from returns up to day t.

Walk-forward protocol (nothing after the forecast day is ever used):
  - the first ``--initial`` days only train; from then on the models are refitted every ``--refit`` days
    and forecast the next ``--refit`` days;
  - training window: all past days (``--window 0``) or the last ``--window`` days;
  - inside each training window the last 20 % choose the hyperparameters (AR order, ESN reservoir, ridge
    strength); the chosen model is then refitted on the whole window;
  - returns are measured in units of the standard deviation of the first ``--initial`` days, so the ESN
    input scale never depends on later data.

Forecasters:
  zero        r[t+1] = 0
  mean        the mean return of the training window (ridge on a constant)
  AR          ridge regression on r[t], ..., r[t-p+1] with an intercept, p in {1, 2, 5, 10, 20}
  ESN n       tanh echo state network with n nodes driven by r (spectral radius 0.5 / 0.8 / 0.95, input
              scale 0.1 / 0.3 / 1, 3 random reservoirs), ridge readout on [states, r[t]] with an intercept

Scores over all forecast days and per decade:
  R2 vs zero  1 - sum (r - forecast)^2 / sum r^2      (> 0: better than predicting zero)
  hit rate    share of days whose sign was forecast right (days with a zero forecast count half)
  DM t        Diebold-Mariano statistic of the squared-error difference against "zero", Newey-West
              standard error with 5 lags (|t| > 2: the difference is unlikely to be chance)

    python experiments/finance_walkforward.py data/fred_nasdaqcom/NASDAQCOM.csv
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
from combrc.evaluate import esn_states  # noqa: E402
from combrc.readout import Ridge  # noqa: E402

LAMBDAS = (1e-4, 1e-2, 1.0, 1e2, 1e4)
AR_ORDERS = (1, 2, 5, 10, 20)
ESN_GRID = [(r, s, e) for r in (0.5, 0.8, 0.95) for s in (0.1, 0.3, 1.0) for e in range(3)]
WARMUP = 100


def load(path):
    """Dates and log returns of a FRED csv (date, value; '.' or empty on holidays)."""
    dates, prices = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            value = list(row.values())[1].strip()
            if value and value != ".":
                dates.append(row[list(row)[0]])
                prices.append(float(value))
    p = np.array(prices)
    return dates[1:], np.diff(np.log(p))


def lags(z, p):
    """[T, p]: z[t], z[t-1], ..., z[t-p+1] (zeros before the start)."""
    out = np.zeros((len(z), p))
    for k in range(p):
        out[k:, k] = z[: len(z) - k]
    return out


def choose_and_fit(features, z, train_idx):
    """Pick (feature set, ridge strength) on the last 20 % of the training rows, refit on all of them.
    ``features``: {name: [T, F] array}. Row t forecasts z[t+1]."""
    n = len(train_idx)
    fit_rows, val_rows = train_idx[: int(0.8 * n)], train_idx[int(0.8 * n):]
    y = torch.tensor(z[train_idx + 1])
    best = (float("inf"), None, None)
    for name, x in features.items():
        xt = torch.tensor(x[fit_rows])
        xv = torch.tensor(x[val_rows])
        for lam in LAMBDAS:
            model = Ridge(lam, eps=1e-6).fit(xt, y[: len(fit_rows)])
            err = float(((model.predict(xv) - y[len(fit_rows):]) ** 2).mean())
            if err < best[0]:
                best = (err, name, lam)
    _, name, lam = best
    return name, Ridge(lam, eps=1e-6).fit(torch.tensor(features[name][train_idx]), y)


def dm_stat(e_ref, e_model, lag=5):
    """Diebold-Mariano t statistic for mean(e_ref^2 - e_model^2) > 0, Newey-West variance."""
    d = e_ref ** 2 - e_model ** 2
    n, mean = len(d), d.mean()
    c = d - mean
    var = c @ c / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (c[k:] @ c[:-k]) / n
    return mean / math.sqrt(var / n) if var > 0 else float("nan")


def scores(z, forecast):
    err = z - forecast
    hit = np.where(forecast == 0, 0.5, (np.sign(forecast) == np.sign(z)).astype(float))
    return {"r2_vs_zero": 1 - (err @ err) / (z @ z), "hit_rate": hit.mean(), "dm_t": dm_stat(z, err)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="FRED csv with a date and a price column")
    ap.add_argument("--initial", type=int, default=2520, help="days used only for training at the start")
    ap.add_argument("--refit", type=int, default=252, help="days between refits (and forecast block length)")
    ap.add_argument("--window", type=int, default=0, help="training window in days (0: all past days)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[31, 100])
    ap.add_argument("--out", default="results/finance_walkforward")
    args = ap.parse_args()
    torch.set_num_threads(2)

    dates, r = load(args.csv)
    scale = r[: args.initial].std()                     # from the first training days only
    z = r / scale
    T = len(z)
    print(f"{len(r)} returns {dates[0]} .. {dates[-1]}; std of the first {args.initial} days {scale:.4f}", flush=True)

    # candidate feature sets, each row t built from z[0..t] only
    feats = {"mean": {"const": np.ones((T, 1))}}         # standardised to zero: the forecast is the intercept
    feats["AR"] = {f"p={p}": lags(z, p) for p in AR_ORDERS}
    for n in args.sizes:
        sets = {}
        for radius, s, e in ESN_GRID:
            states = esn_states(torch.tensor(0.5 + 0.5 * z), n, radius, s, seed=e).numpy()
            sets[f"r={radius} s={s} seed={e}"] = np.hstack([states, z[:, None]])
        feats[f"ESN {n}"] = sets
        print(f"ESN {n}: {len(sets)} reservoirs simulated", flush=True)

    # walk forward: forecast rows t (predicting z[t+1]) for t in [start, start + refit)
    forecast = {"zero": np.zeros(T - 1)}
    forecast.update({k: np.full(T - 1, np.nan) for k in feats})
    chosen = {k: [] for k in feats}
    first = max(args.initial, WARMUP + 1)
    for start in range(first, T - 1, args.refit):
        lo = WARMUP if not args.window else max(WARMUP, start - args.window)
        train_idx = np.arange(lo, start - 1)            # row t uses target z[t+1], so t + 1 < start
        test_idx = np.arange(start, min(start + args.refit, T - 1))
        for k, sets in feats.items():
            name, model = choose_and_fit(sets, z, train_idx)
            forecast[k][test_idx] = model.predict(torch.tensor(sets[name][test_idx])).numpy()
            chosen[k].append(name)
    rows = np.arange(first, T - 1)
    target = z[rows + 1]
    years = np.array([int(dates[t + 1][:4]) for t in rows])

    periods = [("all", np.ones(len(rows), bool))]
    for y0 in range(1980, 2030, 10):
        mask = (years >= y0) & (years < y0 + 10)
        if mask.sum() > 200:
            periods.append((f"{y0}s", mask))
    table = []
    for k in forecast:
        for label, mask in periods:
            s = scores(target[mask], forecast[k][rows][mask])
            table.append({"model": k, "period": label, "days": int(mask.sum()), **s})
    up = (target > 0).mean()
    print(f"\nforecast days {dates[first + 1]} .. {dates[-1]} ({len(rows)}); share of up days {up:.3f}")
    print(f"{'model':8s} {'period':6s} {'days':>6s} {'R2 vs zero':>11s} {'hit rate':>9s} {'DM t':>6s}")
    for row in table:
        print(f"{row['model']:8s} {row['period']:6s} {row['days']:6d} {row['r2_vs_zero']:11.4f} "
              f"{row['hit_rate']:9.3f} {row['dm_t']:6.2f}")
    for k in ("AR", *[f"ESN {n}" for n in args.sizes]):
        counts = {}
        for name in chosen[k]:
            counts[name] = counts.get(name, 0) + 1
        print(f"{k}: chosen " + ", ".join(f"{name} x{c}" for name, c in sorted(counts.items(), key=lambda kv: -kv[1])[:4]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    suffix = "" if not args.window else f"_window{args.window}"
    with open(args.out + suffix + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    np.savez_compressed(args.out + suffix + "_forecasts.npz", dates=np.array([dates[t + 1] for t in rows]),
                        target=target, scale=scale, models=np.array(list(forecast)),
                        forecasts=np.stack([forecast[k][rows] for k in forecast]))
    print("saved", args.out + suffix + ".csv and _forecasts.npz")


if __name__ == "__main__":
    main()
