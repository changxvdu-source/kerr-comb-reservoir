"""Feature preparation and linear (ridge) readout. Everything here is differentiable."""
import math

import torch

LAMBDAS = (1e-8, 1e-6, 1e-4, 1e-2, 1.0)


def supergaussian_filter(n_modes, fsr, width, order=4, dtype=torch.float64):
    """Power transmission exp(-(f / (width/2))^(2*order)) per comb line, pump centred.

    The paper states "super-Gaussian filter of order 4, and a width of 1 THz"
    without defining width; here it is taken as the full width, an assumption.
    """
    f = (torch.arange(n_modes, dtype=dtype) - n_modes // 2) * fsr
    return torch.exp(-((f / (width / 2)) ** (2 * order)))


def select_lines(spectra, n_lines):
    """Keep the ``n_lines`` comb lines centred on the pump."""
    c = spectra.shape[-1] // 2
    return spectra[..., c - n_lines // 2: c + (n_lines + 1) // 2]


def to_features(spectra, log=True, floor=1e-9):
    """Line powers in watts -> regression features (dB-like scale by default).

    ``floor`` plays the role of a detector noise floor (1 nW, about 67 dB below
    the pump line at the drop port). Without it, lines that carry no light turn
    into large-variance features after the logarithm and dominate the regression.
    """
    return torch.log10(spectra + floor) if log else spectra


def add_taps(x, taps):
    """Give the readout memory: append the features of the previous ``taps - 1`` symbols.

    x: [..., T, F] -> [..., T, F * taps]. Row t holds x[t], x[t-1], ..., x[t-taps+1];
    before the start of the record the first row is repeated.
    """
    if taps <= 1:
        return x
    t = torch.arange(x.shape[-2])
    return torch.cat([x[..., (t - k).clamp_min(0), :] for k in range(taps)], dim=-1)


PHOTON_ENERGY_1550 = 6.626_070_15e-34 * 299_792_458 / 1550e-9      # J


def detector_floor(detector_noise=0.0, preamp_gain_db=0.0):
    """Floor for ``to_features`` after ``measure``: the electronic noise referred to the
    optical input, but at least 1 nW, so the logarithm does not amplify readings below the noise."""
    return max(1e-9, detector_noise * 10 ** (-preamp_gain_db / 10))


def _randn_like(x, generator):
    """Standard normal numbers shaped like ``x`` on its device; a CPU generator draws on the CPU, so a
    seeded run sees the same noise on every device."""
    if x.device.type != "cpu" and (generator is None or generator.device.type != "cpu"):
        return torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)
    return torch.randn(x.shape, generator=generator, dtype=x.dtype).to(x.device)


def measure(spectra, detector_noise=0.0, generator=None, shot_time=0.0, preamp_gain_db=0.0,
            preamp_nf_db=0.0, photon_energy=PHOTON_ENERGY_1550):
    """Photodetection of each comb line. Returns the readings as optical power at the drop port (watts).

    Independent Gaussian noise is added to every reading, with variance

        F * P * photon_energy / shot_time  +  (detector_noise / G)^2

    ``detector_noise``  rms electronic noise of the receiver, as equivalent optical power at the detector.
    ``shot_time``       integration time of one reading in seconds; > 0 switches on photon shot noise
                        (the first term with F = 1; the noise bandwidth is 1 / (2 * shot_time)).
    ``preamp_gain_db``, ``preamp_nf_db``   power gain G and noise figure F of an optical amplifier in
                        front of the detectors. Referred to its input, an amplifier turns the shot-noise
                        variance into F times that value, with F = (1 + 2 n_sp (G - 1)) / G, and makes
                        the electronic noise G times less important.

    Not modelled: detector quantum efficiency below 1, and the beat noise of the amplifier's spontaneous
    emission with itself (a few nW rms when every line is filtered to its 200 GHz channel).

    Follow with ``to_features(..., floor=detector_floor(detector_noise, preamp_gain_db))``.
    """
    gain, nf = 10 ** (preamp_gain_db / 10), 10 ** (preamp_nf_db / 10)
    if nf < (2 - 1 / gain) * (1 - 1e-9):
        raise ValueError(f"a {preamp_gain_db:g} dB amplifier cannot have a noise figure below "
                         f"{10 * math.log10(2 - 1 / gain):.2f} dB")
    if shot_time < 0 or (nf > 1 and not shot_time):
        raise ValueError("amplifier noise needs shot_time > 0, the integration time of one reading")
    electronic = detector_noise / gain
    if not shot_time:
        if not electronic:
            return spectra
        noise = _randn_like(spectra, generator)
        return (spectra + electronic * noise).clamp_min(0)
    variance = spectra.clamp_min(0) * (nf * photon_energy / shot_time) + electronic ** 2
    noise = _randn_like(spectra, generator)
    return (spectra + torch.sqrt(variance.clamp_min(1e-40)) * noise).clamp_min(0)     # clamp: finite gradient at P = 0


class Ridge:
    """Ridge regression with standardised features and an unpenalised intercept.

    ``eps`` is added to each feature's standard deviation, so features that barely
    move (here: by less than ~1e-3 decades of optical power) are not blown up to
    unit variance. Use a smaller ``eps`` for features that are not O(1).
    """

    def __init__(self, lam=1e-6, eps=1e-3):
        self.lam = lam
        self.eps = eps

    def standardise(self, x):
        return (x - self.mean) / self.std

    def fit(self, x, y):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, keepdim=True) + self.eps
        z = self.standardise(x)
        self.y_mean = y.mean()
        eye = torch.eye(z.shape[1], dtype=z.dtype, device=z.device)
        self.w = torch.linalg.solve(z.T @ z / len(z) + self.lam * eye, z.T @ (y - self.y_mean) / len(z))
        return self

    def predict(self, x):
        return self.standardise(x) @ self.w + self.y_mean


def nmse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()


def fit_readout(x, y, n_train, n_val, lambdas=LAMBDAS):
    """Fit on the first ``n_train`` rows, pick the ridge strength on the next ``n_val``.

    Returns the chosen readout; rows after ``n_train + n_val`` are never looked at,
    so they can serve as the test set.
    """
    best, best_err = None, float("inf")
    for lam in lambdas:
        model = Ridge(lam).fit(x[:n_train], y[:n_train])
        err = nmse(y[n_train:n_train + n_val], model.predict(x[n_train:n_train + n_val])).item()
        if err < best_err:
            best, best_err = model, err
    return best


def consistency(a, b):
    """Mean per-feature correlation between two runs' features ([T, F] each).

    1 means the response is fully determined by the input; 0 means it is dominated
    by the (noise-dependent) internal dynamics. Features that do not vary are skipped.
    """
    a, b = a - a.mean(0), b - b.mean(0)
    na, nb = a.norm(dim=0), b.norm(dim=0)
    live = (na > 1e-9) & (nb > 1e-9)
    if not live.any():
        return 1.0
    return ((a * b).sum(0)[live] / (na[live] * nb[live])).mean().item()
