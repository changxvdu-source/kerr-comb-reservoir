"""Fibre-link channel. Run: python -m pytest tests -q"""
import math

import numpy as np

from combrc.fiber import beta2_from_d, fiber_pam4, propagate


def test_beta2_of_standard_fibre():
    assert abs(beta2_from_d(17.0) * 1e27 - (-21.7)) < 0.1          # ps^2/km = 1e-27 s^2/m


def test_gaussian_pulse_broadens_as_predicted():
    dt, t0, length = 0.5e-12, 10e-12, 20e3
    t = (np.arange(4096) - 2048) * dt
    field = np.exp(-t ** 2 / (2 * t0 ** 2)).astype(complex)
    b2 = beta2_from_d(17.0)
    out = propagate(field, dt, length, b2, gamma=0.0, alpha_db_km=0.0)
    width = math.sqrt(np.sum(t ** 2 * np.abs(out) ** 2) / np.sum(np.abs(out) ** 2))
    expected = t0 / math.sqrt(2) * math.sqrt(1 + (length * abs(b2) / t0 ** 2) ** 2)    # rms width of |E|^2
    assert abs(width / expected - 1) < 1e-3


def test_self_phase_modulation_without_dispersion():
    dt, length, gamma = 1e-12, 10e3, 1.3e-3
    power = 0.01 * (1 + 0.5 * np.sin(2 * np.pi * np.arange(1024) / 1024))
    out = propagate(np.sqrt(power).astype(complex), dt, length, 0.0, gamma, alpha_db_km=0.0)
    assert np.allclose(np.abs(out) ** 2, power, rtol=1e-9)
    assert np.allclose(np.angle(out * np.exp(-1j * gamma * power * length)), 0, atol=1e-6)


def test_power_fading_of_small_signal_intensity_modulation():
    """Chromatic dispersion turns a weak intensity tone at f into cos(beta2 (2 pi f)^2 L / 2) of itself."""
    n, dt, length = 8192, 1e-12, 20e3
    b2 = beta2_from_d(17.0)
    for f in (5e9, 10e9, 13e9):
        k = round(f * n * dt)
        f = k / (n * dt)
        m = 1e-3
        power = 1 + m * np.cos(2 * np.pi * f * np.arange(n) * dt)
        out = propagate(np.sqrt(power).astype(complex), dt, length, b2, 0.0, alpha_db_km=0.0)
        detected = np.abs(out) ** 2
        tone = 2 * abs(np.fft.fft(detected)[k]) / n
        expected = m * abs(math.cos(b2 * (2 * math.pi * f) ** 2 * length / 2))
        assert abs(tone - expected) < 0.02 * m


def test_task_shapes_and_targets():
    u, y = fiber_pam4(500, seed=1, length_km=5)
    assert u.shape == (500,) and y.shape == (500,)
    assert u.min() == 0 and u.max() == 1
    assert set(np.unique(y)) <= {-3.0, -1.0, 1.0, 3.0}
    u2, y2 = fiber_pam4(500, seed=1, length_km=5)
    assert np.array_equal(u, u2) and np.array_equal(y, y2)
