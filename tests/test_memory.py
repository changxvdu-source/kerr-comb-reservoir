"""Delayed input and optical feedback loop. Run: python -m pytest tests -q"""
import cmath

import torch

from combrc import KerrRing
from combrc.lle import stack_rings


def same(a, b, rtol=1e-9):
    return torch.allclose(a, b, rtol=rtol, atol=1e-12 * b.abs().max().item())


def test_linear_steady_state_with_feedback_loop():
    """Weak CW pump, loop transmission c: E = sqrt(theta1 P) / (alpha + i delta0 - sqrt(theta1 theta2) c)."""
    c = 0.6 * cmath.exp(0.7j)
    ring = KerrRing(n_modes=16, steps_per_symbol=20, p_avg=1e-6, p_swing=0.0, delta0=3e-4).set_feedback(c)
    u = torch.full((1, 3000), 0.5, dtype=torch.float64)
    _, e = ring(u, state=torch.zeros(1, 16, dtype=torch.complex128))
    a, d = ring.alpha.item(), ring.delta0.item()
    t1, t2 = ring.theta1.item(), ring.theta2.item()
    expected = (t1 * 1e-6) ** 0.5 / complex(a, d) / (1 - (t1 * t2) ** 0.5 * c / complex(a, d))
    assert abs(e[0, 0].item() - expected) < 1e-3 * abs(expected)


def test_zero_feedback_and_zero_delay_weight_change_nothing():
    kw = {"n_modes": 32, "steps_per_symbol": 8, "input_modes": [-1, 1], "p_avg": 0.05, "seed": 3}
    u = torch.rand(1, 30, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    plain, _ = KerrRing(**kw)(u)
    other, _ = KerrRing(**kw).set_feedback(0)(u, u2=torch.rand(1, 30, dtype=torch.float64))
    assert same(other, plain)


def test_delayed_input_acts_like_a_second_signal():
    kw = {"n_modes": 32, "steps_per_symbol": 8, "input_modes": [-1, 1], "p_avg": 0.05, "seed": 3}
    u = torch.rand(1, 30, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    ring = KerrRing(**kw).set_delay_input(torch.tensor([0.05, 0.03j]))
    a, _ = ring(u, u2=u)
    b, _ = ring(u, u2=1 - u)
    assert not torch.allclose(a, b)


def test_loop_and_delay_survive_blocks_and_stacking():
    kw = {"n_modes": 32, "steps_per_symbol": 8, "input_modes": [-2, -1, 1, 2]}
    rings = [KerrRing(q_ext1=2e6, delta0=5e-4, p_avg=0.05, seed=1, **kw).set_feedback(0.5j),
             KerrRing(q_ext1=1e6, delta0=-2e-4, p_avg=0.03, seed=2, **kw).set_delay_input(
                 torch.tensor([0.02, 0.01j, -0.03, 0.01]))]
    gen = torch.Generator().manual_seed(0)
    u = torch.rand(2, 40, dtype=torch.float64, generator=gen)
    u2 = torch.rand(2, 40, dtype=torch.float64, generator=gen)
    states = [ring.init_state(1, seed=i) for i, ring in enumerate(rings)]
    alone = torch.cat([ring(u[i:i + 1], state=states[i], u2=u2[i:i + 1])[0] for i, ring in enumerate(rings)])
    stack = stack_rings(rings)
    state, loop, parts = torch.cat(states), None, []
    for t0 in range(0, 40, 15):        # in blocks, carrying the field and the loop
        out, state = stack(u[:, t0:t0 + 15], state=state, u2=u2[:, t0:t0 + 15], loop=loop)
        loop = stack.last_loop
        parts.append(out)
    assert same(torch.cat(parts, 1), alone)


def test_supplied_rin_samples_are_shared_and_only_rin_is():
    """Two different rings stacked, two runs each: with the RIN samples supplied per run, runs with the same
    samples see the same intensity noise; with RIN as the only noise, a ring's two runs given the same
    samples are identical and different samples give different outputs."""
    kw = {"n_modes": 32, "steps_per_symbol": 8, "input_modes": [-1, 1], "p_avg": 0.05}
    rings = [KerrRing(**kw, seed=3), KerrRing(**kw, seed=4)]
    u = torch.rand(1, 30, dtype=torch.float64, generator=torch.Generator().manual_seed(0)).expand(4, -1)
    stack = stack_rings(rings, repeats=2).set_noise(rin=0.05)
    trace = torch.randn(2, 30, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    state = torch.zeros(4, 32, dtype=torch.complex128)
    same_runs, _ = stack(u, state=state, rin=trace[[0, 0]].repeat(2, 1))
    assert same(same_runs[1], same_runs[0]) and same(same_runs[3], same_runs[2])
    two_runs, _ = stack(u, state=state, rin=trace.repeat(2, 1))
    assert same(two_runs[0], same_runs[0]) and same(two_runs[2], same_runs[2])
    assert not torch.allclose(two_runs[1], two_runs[0])
    clean, _ = stack(u, state=state, rin=torch.zeros(4, 30, dtype=torch.float64))
    quiet, _ = stack_rings(rings, repeats=2)(u, state=state)
    assert same(clean, quiet)
