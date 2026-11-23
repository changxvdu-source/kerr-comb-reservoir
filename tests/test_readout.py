"""Checks of the photodetection model. Run: python -m pytest tests -q"""
import math

import pytest
import torch

from combrc.readout import PHOTON_ENERGY_1550, detector_floor, measure


def readings(power, **kw):
    spectra = torch.full((400_000,), power, dtype=torch.float64)
    return measure(spectra, generator=torch.Generator().manual_seed(0), **kw)


def test_no_noise_by_default():
    spectra = torch.rand(2, 5, 7, dtype=torch.float64)
    assert measure(spectra) is spectra


def test_shot_noise_is_poissonian():
    """Over 1 ns, 10 uW is 78 000 photons, so the relative rms must be 1 / sqrt(78 000)."""
    power, time = 1e-5, 1e-9
    photons = power * time / PHOTON_ENERGY_1550
    out = readings(power, shot_time=time)
    assert abs(out.mean() / power - 1) < 1e-4
    assert abs(out.std() / power * math.sqrt(photons) - 1) < 0.01


def test_preamplifier_scales_shot_noise_and_suppresses_detector_noise():
    power, time, detector = 1e-5, 1e-9, 3e-7
    shot = power * PHOTON_ENERGY_1550 / time                      # variance without amplifier
    direct = readings(power, detector_noise=detector, shot_time=time)
    assert abs(direct.var() / (shot + detector ** 2) - 1) < 0.01
    amplified = readings(power, detector_noise=detector, shot_time=time, preamp_gain_db=20, preamp_nf_db=5)
    expected = 10 ** 0.5 * shot + (detector / 100) ** 2
    assert abs(amplified.var() / expected - 1) < 0.01
    assert amplified.var() < direct.var() / 10                    # here the detector noise dominated


def test_detector_noise_alone_is_unchanged_by_the_new_terms():
    out = readings(1e-5, detector_noise=3e-7)
    assert abs(out.std() / 3e-7 - 1) < 0.01


def test_unphysical_amplifier_is_rejected():
    with pytest.raises(ValueError):
        readings(1e-5, shot_time=1e-9, preamp_gain_db=20, preamp_nf_db=2)     # below the 3 dB quantum limit
    with pytest.raises(ValueError):
        readings(1e-5, preamp_gain_db=20, preamp_nf_db=5)                      # no integration time given


def test_floor_is_referred_to_the_optical_input():
    assert detector_floor() == 1e-9
    assert detector_floor(3e-7) == 3e-7
    assert detector_floor(3e-7, preamp_gain_db=20) == pytest.approx(3e-9)


def test_gradient_is_finite_for_dark_lines():
    spectra = torch.tensor([0.0, 1e-6], dtype=torch.float64, requires_grad=True)
    measure(spectra, shot_time=1e-9, generator=torch.Generator().manual_seed(0)).sum().backward()
    assert torch.isfinite(spectra.grad).all()
