"""Chaotic benchmark time series (Mackey-Glass, Lorenz, Rossler), rescaled to [0, 1].

The paper gives the equations and parameters but not the sampling interval of the
symbols, so the ``dt`` defaults below are assumptions (standard choices in the
reservoir-computing literature).
"""
import functools

import numpy as np


def _rk4(f, x, h):
    k1 = f(x)
    k2 = f(x + h / 2 * k1)
    k3 = f(x + h / 2 * k2)
    k4 = f(x + h * k3)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def mackey_glass(n, dt=1.0, beta=0.2, gamma=0.1, tau=17.0, power=10, sub=10, discard=1000, seed=0):
    h = dt / sub
    lag = int(round(tau / h))
    rng = np.random.default_rng(seed)
    total = (n + discard) * sub
    x = np.empty(total + lag)
    x[:lag] = 1.2 + 0.1 * rng.standard_normal(lag)

    def f(xt, xd):
        return beta * xd / (1 + xd ** power) - gamma * xt

    for i in range(lag, total + lag):
        # RK4 with the delayed term held piecewise constant over the sub-step.
        xd = x[i - lag]
        xt = x[i - 1]
        k1 = f(xt, xd)
        k2 = f(xt + h / 2 * k1, xd)
        k3 = f(xt + h / 2 * k2, xd)
        k4 = f(xt + h * k3, xd)
        x[i] = xt + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x[lag + discard * sub::sub][:n]


def _ode_series(f, x0, n, dt, sub, discard, seed):
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float) + 0.1 * rng.standard_normal(len(x0))
    h = dt / sub
    out = np.empty(n)
    for i in range(n + discard):
        for _ in range(sub):
            x = _rk4(f, x, h)
        if i >= discard:
            out[i - discard] = x[0]
    return out


def lorenz(n, dt=0.02, sigma=10.0, rho=28.0, beta=8.0 / 3.0, sub=10, discard=1000, seed=0):
    def f(s):
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
    return _ode_series(f, [1.0, 1.0, 20.0], n, dt, sub, discard, seed)


def rossler(n, dt=0.1, a=0.2, b=0.2, c=5.7, sub=10, discard=1000, seed=0):
    def f(s):
        x, y, z = s
        return np.array([-y - z, x + a * y, b + z * (x - c)])
    return _ode_series(f, [1.0, 1.0, 0.0], n, dt, sub, discard, seed)


SERIES = {"mackey_glass": mackey_glass, "rossler": rossler, "lorenz": lorenz}


def make_series(name, n, seed=0, **kw):
    """Return ``n`` symbols of the named series, linearly rescaled to [0, 1]."""
    x = SERIES[name](n, seed=seed, **kw)
    return (x - x.min()) / (x.max() - x.min())


# ---- supervised benchmark tasks: an input sequence u in [0, 1] and an aligned target y ----

def narma10(n, seed=0, discard=200):
    """NARMA-10: y[t+1] = 0.3 y[t] + 0.05 y[t] sum(y[t-9..t]) + 1.5 u[t-9] u[t] + 0.1, u ~ U(0, 0.5).

    Returns the input rescaled to [0, 1] and the target aligned so that the model,
    having just seen u[t], must output y[t+1].
    """
    rng = np.random.default_rng(seed)
    total = n + discard + 1
    u = rng.uniform(0, 0.5, total)
    y = np.zeros(total + 1)
    for t in range(9, total):
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * y[t - 9: t + 1].sum() + 1.5 * u[t - 9] * u[t] + 0.1
    return u[discard: discard + n] / 0.5, y[discard + 1: discard + n + 1]


CHANNEL_SYMBOLS = np.array([-3.0, -1.0, 1.0, 3.0])


def channel_equalization(n, seed=0, snr_db=28.0):
    """Nonlinear channel equalisation (Jaeger & Haas 2004): recover the symbol d[t-2] from the
    received signal u[0..t], which is a linear mix of neighbouring symbols passed through a
    polynomial nonlinearity plus noise. Returns (received signal rescaled to [0, 1], d[t-2])."""
    rng = np.random.default_rng(seed)
    pad = 10
    d = rng.choice(CHANNEL_SYMBOLS, n + 2 * pad)
    taps = {2: 0.08, 1: -0.12, 0: 1.0, -1: 0.18, -2: -0.1, -3: 0.091, -4: -0.05, -5: 0.04, -6: 0.03, -7: 0.01}
    idx = np.arange(pad, pad + n)
    q = sum(c * d[idx + k] for k, c in taps.items())
    clean = q + 0.036 * q ** 2 - 0.011 * q ** 3
    received = clean + rng.normal(0, np.sqrt(clean.var() / 10 ** (snr_db / 10)), n)
    received = (received - received.min()) / (received.max() - received.min())
    return received, d[idx - 2]


def is_symbol_task(name):
    """Tasks whose targets are PAM-4 / channel symbols, scored also by the symbol error rate."""
    return name.startswith("channel_eq") or name.startswith("fiber_pam4")


def symbol_error_rate(y_true, y_pred):
    """Fraction of predictions whose nearest channel symbol differs from the true one."""
    nearest = CHANNEL_SYMBOLS[np.abs(np.asarray(y_pred)[:, None] - CHANNEL_SYMBOLS[None, :]).argmin(1)]
    return float((nearest != np.asarray(y_true)).mean())


@functools.lru_cache(maxsize=16)
def _fiber_task(name, n, seed):
    from .fiber import fiber_pam4
    length = float(name[len("fiber_pam4_"):-2]) if name.endswith("km") else 20.0
    return fiber_pam4(n, seed, length_km=length)


def make_task(name, n, seed=0):
    """Return (u, y): n input symbols in [0, 1] and the target for each of them.

    ``name`` is "narma10", "channel_eq", "fiber_pam4[_<L>km]" (PAM-4 over an IM/DD fibre link, see
    fiber.py), or "<series>_h<k>" for predicting a chaotic
    series k steps ahead, e.g. "mackey_glass_h10".
    """
    if name == "narma10":
        return narma10(n, seed)
    if name == "channel_eq":
        return channel_equalization(n, seed)
    if name.startswith("channel_eq_snr"):       # the same channel with another noise level, e.g. channel_eq_snr40
        return channel_equalization(n, seed, snr_db=float(name[len("channel_eq_snr"):]))
    if name.startswith("fiber_pam4"):           # "fiber_pam4" (20 km) or "fiber_pam4_<length>km"
        u, y = _fiber_task(name, n, seed)        # the split-step simulation is cached: many jobs share a series
        return u.copy(), y.copy()
    series, _, horizon = name.rpartition("_h")
    if series not in SERIES:
        raise ValueError(f"unknown task {name!r}")
    x = make_series(series, n + int(horizon), seed=seed)
    return x[:n], x[int(horizon):]
