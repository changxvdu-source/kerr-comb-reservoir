"""One evaluation protocol for the ring and for the software baselines.

Task: from what the model has seen up to symbol t, predict u[t + horizon].
Split of the ``n`` scored symbols: 60 % fit the readout, 20 % choose the ridge
strength, 20 % test. Only the test NMSE is reported.
"""
import torch

from .readout import (add_taps, consistency, detector_floor, fit_readout, measure, nmse, select_lines,
                      to_features)

WARMUP, N_SYMBOLS = 300, 1700


def _splits(n, n_test=None):
    """(training, validation) sizes for n scored symbols: 60/20/20 by default; with ``n_test`` the last
    ``n_test`` symbols are the test part and the rest is split 3:1, so that a series of
    0.8 * N_SYMBOLS + n_test symbols keeps the training and validation parts of N_SYMBOLS."""
    if n_test is None:
        return int(0.6 * n), int(0.2 * n)
    fit = n - n_test
    if fit <= 0:
        raise ValueError(f"n_test={n_test} leaves nothing to fit on in {n} symbols")
    return int(0.75 * fit), fit - int(0.75 * fit)


def held_out(x, y, x_test=None, n_test=None):
    """Fit/select on the first 80 % of (x, y) and predict with features ``x_test`` (default: x).

    Returns (validation NMSE, test NMSE, test predictions): the validation number is
    from the middle 20 %, for choosing between models; the test number from the last 20 %.
    ``n_test``: a test part of this many symbols instead, see ``_splits``.
    """
    n_train, n_val = _splits(len(y), n_test)
    model = fit_readout(x, y, n_train, n_val)
    pred = model.predict(x if x_test is None else x_test)
    val = nmse(y[n_train:n_train + n_val], pred[n_train:n_train + n_val]).item()
    return val, nmse(y[n_train + n_val:], pred[n_train + n_val:]).item(), pred[n_train + n_val:]


def held_out_nmse(x, y, x_test=None, n_test=None):
    """Test NMSE of ``held_out``."""
    return held_out(x, y, x_test, n_test)[1]


def targets(u, horizon, warmup=WARMUP, n=N_SYMBOLS):
    """Inputs are u[0 : warmup+n]; the target for input index t is u[t + horizon]."""
    return u[warmup + horizon: warmup + horizon + n]


@torch.no_grad()
def simulate(ring, u, warmup=WARMUP, n=N_SYMBOLS, seeds=(0, 1), state=None, u2=None):
    """Drop-port spectra [len(seeds), n, n_modes] of independent runs on the same input.

    The runs differ in the seed noise of the initial field and, if noise was enabled
    with ``ring.set_noise``, in every noise sample along the way. ``state`` overrides
    the initial fields (one row per seed), e.g. to start from a soliton. ``u2``: the
    delayed signal for a ring with a delay line (see ``KerrRing.forward``).
    """
    generator = torch.Generator().manual_seed(7919 * seeds[0] + seeds[-1])
    spectra, _ = ring(u[: warmup + n].expand(len(seeds), -1),
                      state=torch.cat([ring.init_state(1, seed=s) for s in seeds]) if state is None else state,
                      generator=generator,
                      u2=None if u2 is None else u2[: warmup + n].expand(len(seeds), -1))
    return spectra[:, warmup:]


def score_spectra(spectra, y, n_lines=51, detector_noise=0.0, seed=0, error_rate=None, taps=1, n_test=None,
                  **detection):
    """Score two runs' spectra (from ``simulate``) against targets ``y``.

    ``detector_noise`` and ``detection`` (shot_time, preamp_gain_db, preamp_nf_db) describe the
    photodetection of the comb lines, see ``readout.measure``.


    ``nmse``        readout trained and tested on the same run;
    ``nmse_cross``  readout trained on one run, tested on the other. This is the
                    number that matters for a physical device, which never sees
                    the same noise twice;
    ``consistency`` correlation between the two runs' features.
    ``val_nmse_cross`` is the same as ``nmse_cross`` on the validation part, for model
    selection. ``error_rate`` (optional) maps (true, predicted) test targets to a rate.
    ``taps`` > 1 lets the readout also see the spectra of the previous taps - 1 symbols.
    ``n_test``: length of the test part, see ``_splits`` (default: the last 20 %).
    """
    generator = torch.Generator().manual_seed(seed)
    lines = measure(select_lines(spectra, n_lines), detector_noise, generator, **detection)
    floor = detector_floor(detector_noise, detection.get("preamp_gain_db", 0.0))
    x = add_taps(to_features(lines, floor=floor), taps)
    val_cross, test_cross, pred = held_out(x[0], y, x_test=x[1], n_test=n_test)
    out = {
        "nmse": held_out_nmse(x[0], y, n_test=n_test),
        "nmse_cross": test_cross,
        "val_nmse_cross": val_cross,
        "consistency": consistency(x[0], x[1]),
    }
    if error_rate is not None:
        out["error_rate_cross"] = error_rate(y[len(y) - len(pred):], pred)
    return out


def score_ring(ring, u, horizon=1, n_lines=51, warmup=WARMUP, n=N_SYMBOLS, seeds=(0, 1), detector_noise=0.0,
               taps=1, **detection):
    """Score a fixed ring on series ``u`` (needs warmup + n + horizon symbols)."""
    spectra = simulate(ring, u, warmup, n, seeds)
    out = score_spectra(spectra, targets(u, horizon, warmup, n), n_lines, detector_noise, taps=taps, **detection)
    last = spectra[0, -200:].mean(0)
    out["comb_lines"] = int((last > last.max() * 1e-6).sum())
    return out


def lagged(u, k, warmup=WARMUP, n=N_SYMBOLS):
    """Rows [u[t], u[t-1], ..., u[t-k+1]] for t = warmup .. warmup+n-1."""
    return torch.stack([u[warmup - j: warmup - j + n] for j in range(k)], 1)


def esn_states(u, n_nodes=200, spectral_radius=0.9, input_scale=1.0, seed=0):
    """States of a standard tanh echo state network driven by ``u`` ([T]) -> [T, n_nodes]."""
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(n_nodes, n_nodes, generator=gen, dtype=torch.float64)
    w = w * (spectral_radius / torch.linalg.eigvals(w).abs().max())
    w_in = input_scale * (2 * torch.rand(n_nodes, generator=gen, dtype=torch.float64) - 1)
    bias = 0.2 * (2 * torch.rand(n_nodes, generator=gen, dtype=torch.float64) - 1)
    x = torch.zeros(n_nodes, dtype=torch.float64)
    out = []
    for value in (2 * u - 1):
        x = torch.tanh(w @ x + w_in * value + bias)
        out.append(x)
    return torch.stack(out)


@torch.no_grad()
def score_baselines(u, horizon=1, warmup=WARMUP, n=N_SYMBOLS, esn_sizes=(51, 200), input_noise=0.0, y=None,
                    taps=1, n_test=None):
    """Test NMSE of input-only baselines under the same protocol as ``score_ring``.

    ``y``: targets aligned with ``u`` (from ``make_task``); default is u[t + horizon].
    ``taps``: the ESN readout also sees the states of the previous taps - 1 symbols.
    ``n_test``: length of the test part, see ``_splits`` (default: the last 20 %).

    ``input_noise``: rms of white noise added to the symbols the baselines see (the
    targets stay clean), to compare with a ring whose input laser is noisy.
    """
    y = targets(u, horizon, warmup, n) if y is None else y[warmup: warmup + n]
    if input_noise:
        gen = torch.Generator().manual_seed(0)
        u = u + input_noise * torch.randn(u.shape, generator=gen, dtype=u.dtype)
    start = int(0.8 * n) if n_test is None else n - n_test
    out = {"persistence": nmse(y[start:], u[warmup: warmup + n][start:]).item()}
    for k in (4, 8, 16, 32):
        out[f"linear_ar{k}"] = held_out_nmse(lagged(u, k, warmup, n), y, n_test=n_test)
    for size in esn_sizes:
        scores = []
        for radius in (0.5, 0.8, 0.95, 1.1):
            for scale in (0.1, 0.3, 1.0):
                states = add_taps(esn_states(u[: warmup + n], size, radius, scale), taps)[warmup:]
                n_train, n_val = _splits(n, n_test)
                model = fit_readout(states, y, n_train, n_val)
                val = nmse(y[n_train:n_train + n_val], model.predict(states[n_train:n_train + n_val])).item()
                test = nmse(y[n_train + n_val:], model.predict(states[n_train + n_val:])).item()
                scores.append((val, test))
        out[f"esn{size}"] = min(scores)[1]      # hyper-parameters chosen on the validation part
    return out
