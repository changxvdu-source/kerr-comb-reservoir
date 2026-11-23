"""Checks of the simulator against analytic results. Run: python -m pytest tests -q"""
import torch

from combrc import KerrRing


def test_cw_steady_state_matches_cubic():
    """Without noise the field stays homogeneous and must settle on the analytic CW solution."""
    ring = KerrRing(n_modes=16, steps_per_symbol=40)
    p_in = torch.tensor([0.05], dtype=torch.float64)       # below the bistable region
    e = torch.zeros(1, 16, dtype=torch.complex128)
    for _ in range(800):                                    # 800 ns >> photon lifetime (~8 ns)
        e = ring.evolve(e, ring.cw_drive(p_in))
    p_cav = (e.abs() ** 2).mean()
    assert torch.allclose(ring.cw_input_power(p_cav), p_in[0], rtol=1e-8)


def test_fourth_order_convergence():
    """Halving the step must shrink the error ~16x (RK4IP is fourth order)."""
    p_in = torch.tensor([0.325], dtype=torch.float64)

    def run(steps):
        ring = KerrRing(n_modes=64, steps_per_symbol=steps)
        e = ring.init_state(1, noise=1e-3, seed=1)
        for _ in range(3):
            e = ring.evolve(e, ring.cw_drive(p_in))
        return e

    ref = run(640)
    err_coarse = (run(20) - ref).abs().max()
    err_fine = (run(40) - ref).abs().max()
    assert 10 < err_coarse / err_fine < 24


def test_total_q_and_photon_lifetime():
    ring = KerrRing()
    assert abs(ring.q_total.item() / 1e7 - 1) < 1e-3
    t_ph = ring.q_total / ring.omega0
    assert 7e-9 < t_ph.item() < 9e-9


def test_mi_gain_positive_on_upper_branch_only():
    ring = KerrRing()
    assert ring.mi_gain(torch.tensor(60.0, dtype=torch.float64)).max() > 0
    assert ring.mi_gain(torch.tensor(1.0, dtype=torch.float64)).max() < 0


def test_gradients_match_finite_differences():
    """Autograd through the solver agrees with central finite differences."""
    u = torch.tensor([[0.2, 0.9, 0.4]], dtype=torch.float64)

    def loss_of(ring):
        spectra, _ = ring(u, state=ring.init_state(1, noise=1e-2, seed=2))
        return torch.log10(spectra).sum()   # log scale: sensitive to the weak sidebands too

    ring = KerrRing(n_modes=32, steps_per_symbol=20, trainable=("delta0", "beta2", "log_q_ext1", "p_avg", "input_weight"),
                    input_modes=(-2, 1, 3))
    loss_of(ring).backward()
    for name in ("_delta0", "_beta2", "_log_q_ext1", "_log_p_avg", "_input_weight"):
        p = getattr(ring, name)
        eps = 1e-6
        with torch.no_grad():
            flat = p.view(-1)                       # check the first element of each parameter
            flat[0] += eps
            up = loss_of(ring)
            flat[0] -= 2 * eps
            down = loss_of(ring)
            flat[0] += eps
        fd = (up - down) / (2 * eps)
        assert torch.allclose(p.grad.view(-1)[0], fd, rtol=1e-4, atol=1e-9), name


def test_input_line_lands_on_its_resonance():
    """Without Kerr effect, a CW line injected at mode mu settles at the linear cavity response."""
    ring = KerrRing(n_modes=32, steps_per_symbol=40, gamma=0.0, q_ext1=1e6, q_ext2=1e6,
                    input_modes=(3,), input_init=0.1)
    with torch.no_grad():
        ring._input_weight.zero_()
    u = torch.full((1,), 0.5, dtype=torch.float64)
    e = torch.zeros(1, 32, dtype=torch.complex128)
    for _ in range(60):                                     # photon lifetime here is ~0.4 ns
        e = ring.evolve(e, ring.drive(u))
    a = torch.fft.fft(e)[0] / 32
    expected = torch.sqrt(ring.theta1) * ring.input_bias[0] / -ring.linear_operator()[3]
    assert torch.allclose(a[3], expected, rtol=1e-8)
    assert a[[1, 2, 4, 5, 29]].abs().max() < 1e-12 * a[3].abs()


def test_identical_noise_free_runs_are_consistent_and_seeds_differ():
    ring = KerrRing(n_modes=32, steps_per_symbol=10)
    state = ring.init_state(3, seed=4)
    assert not torch.allclose(state[0], state[1])
    u = torch.rand(1, 5, dtype=torch.float64).expand(3, -1)
    spectra, _ = ring(u, state=state[:1].expand(3, -1).clone())
    assert torch.allclose(spectra[0], spectra[2])


def test_field_noise_reaches_analytic_variance():
    """Undriven linear cavity: dA = -alpha A dn + field dW  =>  <|A|^2> = field^2 / (2 alpha)."""
    ring = KerrRing(n_modes=32, steps_per_symbol=40, gamma=0.0, q_ext1=1e6, q_ext2=1e6)
    field = 1e-3
    gen = torch.Generator().manual_seed(0)
    e = torch.zeros(256, 32, dtype=torch.complex128)
    drive = ring.cw_drive(torch.zeros(256, dtype=torch.float64))
    for _ in range(8):                                      # 8 ns, many photon lifetimes (~0.4 ns)
        e = ring.evolve(e, drive, field_noise=field, generator=gen)
    measured = (torch.fft.fft(e) / 32).abs().pow(2).mean()
    expected = field ** 2 / (2 * ring.alpha)
    assert abs(measured / expected - 1) < 0.08


def test_noise_is_off_by_default_and_differs_per_row_when_on():
    u = torch.rand(1, 6, dtype=torch.float64).expand(2, -1)
    ring = KerrRing(n_modes=32, steps_per_symbol=10, q_ext1=1e6)
    state = ring.init_state(1).expand(2, -1).clone()
    clean, _ = ring(u, state=state)
    assert torch.allclose(clean[0], clean[1])
    for kw in ({"rin": 1e-2}, {"detuning": 0.1}, {"detuning": 0.1, "detuning_corr": 50.0}, {"field": 1e-4}):
        noisy, _ = ring.set_noise(**kw)(u, state=state, generator=torch.Generator().manual_seed(1))
        assert not torch.allclose(noisy[0], noisy[1]), kw
        again, _ = ring(u, state=state, generator=torch.Generator().manual_seed(1))
        assert torch.allclose(noisy, again), kw             # same generator seed -> same run
    ring.set_noise()
    assert torch.allclose(ring(u, state=state)[0], clean)


def test_quantum_noise_level_is_tiny():
    ring = KerrRing(q_ext1=1e6)
    assert 1e-6 < ring.quantum_field_noise() < 1e-4        # sqrt(W) per mode per round trip


def test_cascade_linear_steady_state_and_feed_forward():
    """Without Kerr effect the second ring settles at link * a1 / (alpha2 + i*delta2); ring 0 is unaffected."""
    from combrc.cascade import RingCascade, follower_ring
    first = KerrRing(n_modes=32, steps_per_symbol=40, gamma=0.0, q_ext1=1e6, q_ext2=1e6, p_swing=0.0)
    second = follower_ring(first, q_ext1=2e6, q_ext2=1e6, delta0=1e-3)
    cascade = RingCascade([first, second], gains_db=[6.0])
    u = torch.full((1, 40), 0.5, dtype=torch.float64)          # 40 ns, many photon lifetimes
    zero = [torch.zeros(1, 32, dtype=torch.complex128) for _ in range(2)]
    spectra, fields = cascade(u, state=zero)
    a1, a2 = (torch.fft.fft(e)[0, 0] / 32 for e in fields)
    assert torch.allclose(a2, cascade.link(1) * a1 / -second.linear_operator()[0], rtol=1e-6)
    alone, _ = first(u, state=zero[0])
    assert torch.allclose(spectra[:, :, 0], alone)
    assert spectra.shape == (1, 40, 2, 32)
