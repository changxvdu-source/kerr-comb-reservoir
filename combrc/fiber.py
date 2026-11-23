"""A short-reach optical fibre link, as a channel-equalisation task for the reservoir.

Intensity-modulation / direct-detection (IM/DD) PAM-4 over standard single-mode fibre at 1550 nm, the
setting of data-centre and access links:

  transmitter   PAM-4 symbols d in {-3, -1, 1, 3} set the optical power of a chirp-free modulator,
                with a finite extinction ratio; the drive is shaped by a Gaussian low-pass filter
                (modulator / driver bandwidth); field E = sqrt(P(t)).
  fibre         split-step Fourier solution of the nonlinear Schroedinger equation
                    dE/dz = -alpha/2 E - i beta2/2 d^2E/dt^2 + i gamma |E|^2 E
                with chromatic dispersion (D = 17 ps/nm/km), loss and the Kerr effect.
  receiver      square-law photodiode, Gaussian electrical low-pass, additive thermal noise at a set
                signal-to-noise ratio, one sample per symbol at the symbol centre.

Dispersion followed by square-law detection is what makes the task hard: the detected signal is a
nonlinear function of neighbouring symbols, and at long enough distance its spectrum has a null
("power fading") that a linear feed-forward equaliser cannot undo.

The time axis is periodic (FFT); the sequence is padded with ``pad`` guard symbols on each side.
"""
import math

import numpy as np

from .tasks import CHANNEL_SYMBOLS

C = 299_792_458.0


def beta2_from_d(d_ps_nm_km, wavelength=1550e-9):
    """Group-velocity dispersion beta2 (s^2/m) from the dispersion parameter D (ps/nm/km)."""
    d = d_ps_nm_km * 1e-6                                   # s/m^2
    return -d * wavelength ** 2 / (2 * math.pi * C)


def gaussian_lowpass(x, dt, bandwidth):
    """Gaussian low-pass with 3 dB bandwidth ``bandwidth`` (Hz) applied along the last axis (periodic)."""
    f = np.fft.fftfreq(x.shape[-1], dt)
    h = np.exp(-math.log(2) / 2 * (f / bandwidth) ** 2)    # |H|^2 = 1/2 at f = bandwidth
    return np.fft.ifft(np.fft.fft(x) * h)


def propagate(field, dt, length, beta2, gamma, alpha_db_km=0.2, step=None):
    """Symmetric split-step Fourier propagation of ``field`` (sqrt(W)) over ``length`` metres."""
    alpha = alpha_db_km / (10 * math.log10(math.e)) / 1e3    # 1/m (power)
    omega = 2 * math.pi * np.fft.fftfreq(field.shape[-1], dt)
    if step is None:                                         # nonlinear phase per step at most ~1 mrad
        peak = float(np.max(np.abs(field) ** 2))
        step = length if gamma * peak == 0 else min(length, 1e-3 / (gamma * peak))
    n = max(1, math.ceil(length / step))
    h = length / n
    half = np.exp((1j * beta2 / 2 * omega ** 2 - alpha / 2) * h / 2)
    e = np.fft.ifft(half * np.fft.fft(field))
    for k in range(n):
        e = e * np.exp(1j * gamma * np.abs(e) ** 2 * h)
        lin = half if k == n - 1 else half * half
        e = np.fft.ifft(lin * np.fft.fft(e))
    return e


def fiber_pam4(n, seed=0, baud=28e9, length_km=20.0, launch_dbm=3.0, extinction_db=8.0, sps=16,
               tx_bandwidth=0.8, rx_bandwidth=0.75, snr_db=24.0, d_ps_nm_km=17.0, gamma=1.3e-3,
               alpha_db_km=0.2, delay=4, pad=64, return_details=False):
    """PAM-4 over an IM/DD fibre link. Returns (received samples rescaled to [0, 1], d[t - delay]).

    ``baud`` symbol rate; ``length_km`` fibre length; ``launch_dbm`` mean launch power; ``sps`` samples per
    symbol of the waveform simulation; ``tx_bandwidth`` and ``rx_bandwidth`` the transmitter and receiver
    3 dB bandwidths as fractions of the symbol rate; ``snr_db`` electrical SNR at the decision point (signal
    variance over noise variance); ``gamma`` in 1/(W m); ``delay`` how many symbols the decision lags the
    received sample, so that the equaliser also sees symbols that follow (dispersion spreads both ways).
    """
    rng = np.random.default_rng(seed)
    total = n + 2 * pad
    d = rng.choice(CHANNEL_SYMBOLS, total)
    t_sym = 1.0 / baud
    dt = t_sym / sps

    # transmitter: power levels with extinction ratio, mean = launch power
    er = 10 ** (extinction_db / 10)
    p_mean = 1e-3 * 10 ** (launch_dbm / 10)
    p_min = 2 * p_mean / (1 + er)
    levels = p_min + (d + 3) / 6 * (er - 1) * p_min          # p_min .. er * p_min
    drive = gaussian_lowpass(np.repeat(levels, sps).astype(complex), dt, tx_bandwidth * baud).real
    field = np.sqrt(np.clip(drive, 0, None)).astype(complex)

    # fibre
    beta2 = beta2_from_d(d_ps_nm_km)
    out = propagate(field, dt, length_km * 1e3, beta2, gamma, alpha_db_km)

    # receiver: photodiode, electrical low-pass, sample at symbol centres, thermal noise
    current = gaussian_lowpass(np.abs(out) ** 2 + 0j, dt, rx_bandwidth * baud).real
    samples = current[sps // 2::sps][:total]
    noise_rms = math.sqrt(np.var(samples) / 10 ** (snr_db / 10))
    samples = samples + noise_rms * rng.standard_normal(total)

    idx = np.arange(pad, pad + n)
    received = samples[idx]
    received = (received - received.min()) / (received.max() - received.min())
    if return_details:
        return received, d[idx - delay], {"field_in": field, "field_out": out, "dt": dt, "beta2": beta2,
                                          "symbols": d, "samples": samples}
    return received, d[idx - delay]
