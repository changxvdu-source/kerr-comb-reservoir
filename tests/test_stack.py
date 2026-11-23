"""Stacked rings and devices. Run: python -m pytest tests -q"""
import pytest
import torch

from combrc import KerrRing
from combrc.lle import stack_rings

NOISE = {"rin": 1e-3, "detuning": 0.02}


def same(a, b, rtol=1e-9):
    """Equal up to rounding; lines that carry no light are compared on the scale of the brightest one."""
    return torch.allclose(a, b, rtol=rtol, atol=1e-12 * b.max().item())


def some_rings():
    kw = {"n_modes": 64, "steps_per_symbol": 10, "input_modes": [-2, -1, 1, 2]}
    return [KerrRing(q_ext1=1e6, delta0=-2e-4, p_avg=0.05, p_swing=0.02, seed=1, **kw),
            KerrRing(q_ext1=5e6, delta0=1e-3, p_avg=0.02, p_swing=0.0, detuning_depth=0.5, seed=2, **kw),
            KerrRing(q_ext1=2e6, q_ext2=1e7, delta0=5e-4, beta2=-8e-27, p_avg=0.1, p_swing=0.01, phase_depth=0.8,
                     seed=3, **kw)]


def test_stacked_rings_match_the_individual_rings():
    rings = some_rings()
    u = torch.rand(3, 40, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    states = [ring.init_state(1, seed=i) for i, ring in enumerate(rings)]
    alone = torch.cat([ring(u[i:i + 1], state=states[i])[0] for i, ring in enumerate(rings)])
    together, _ = stack_rings(rings)(u, state=torch.cat(states))
    assert same(together, alone)


def test_repeats_give_every_ring_several_rows():
    rings = some_rings()
    u = torch.rand(1, 20, dtype=torch.float64, generator=torch.Generator().manual_seed(0)).expand(6, -1)
    state = torch.cat([ring.init_state(2, seed=7) for ring in rings])
    together, _ = stack_rings(rings, repeats=2)(u, state=state)
    for i, ring in enumerate(rings):
        alone, _ = ring(u[:2], state=state[2 * i: 2 * i + 2])
        assert same(together[2 * i: 2 * i + 2], alone)


def test_stack_of_identical_rings_sees_the_same_noise_as_a_batch():
    ring = some_rings()[0]
    u = torch.rand(3, 20, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    state = ring.init_state(3, seed=5)
    ring.set_noise(field=30 * ring.quantum_field_noise(), **NOISE)
    batch, _ = ring(u, state=state, generator=torch.Generator().manual_seed(1))
    stack = stack_rings([ring] * 3)
    stack.set_noise(field=30 * stack.quantum_field_noise(), **NOISE)
    together, _ = stack(u, state=state, generator=torch.Generator().manual_seed(1))
    assert same(together, batch)
    clean, _ = ring.set_noise()(u, state=state)
    assert not torch.allclose(batch, clean, rtol=1e-6)


def test_rings_with_different_grids_cannot_be_stacked():
    with pytest.raises(ValueError):
        stack_rings([KerrRing(n_modes=64), KerrRing(n_modes=32)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_gpu_matches_cpu():
    rings = some_rings()
    u = torch.rand(3, 40, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    state = torch.cat([ring.init_state(1, seed=i) for i, ring in enumerate(rings)])
    cpu = stack_rings(rings)
    gpu = stack_rings(rings).to("cuda")
    for model in (cpu, gpu):
        model.set_noise(field=30 * model.quantum_field_noise(), **NOISE)
    a, _ = cpu(u, state=state, generator=torch.Generator().manual_seed(1))       # CPU generator: same noise
    b, _ = gpu(u.to("cuda"), state=state.to("cuda"), generator=torch.Generator().manual_seed(1))
    assert b.is_cuda and same(b.cpu(), a, rtol=1e-8)
    c, _ = gpu(u.to("cuda"), state=state.to("cuda"), generator=torch.Generator(device="cuda").manual_seed(1))
    assert not torch.allclose(c.cpu(), a, rtol=1e-6)          # drawn on the GPU: a different realisation
    total = c.cpu().sum(-1) / a.sum(-1)                        # of the same small noise: equal total power
    assert (total - 1).abs().max() < 0.05
