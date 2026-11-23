"""Rings in series: the drop-port field of each ring drives the next one, all-optically.

    signal -> [ring 0] --drop--> (gain) --> [ring 1] --drop--> (gain) --> ...

The link is coherent: the complex field leaving the drop port of ring k-1, amplified
by a power gain G_k, is the input field of ring k,

    drive_k = sqrt(theta1_k * G_k * theta2_{k-1}) * E_{k-1}

so no detection or re-modulation happens between the rings. The rings must share the
free spectral range (their resonances have to line up with the incoming comb lines),
which in practice means thermal tuning; each ring keeps its own Q, detuning and
dispersion. Light only travels forward, so ring 0 behaves exactly as it does alone.

Noise sources (``KerrRing.set_noise``) are not applied inside a cascade.
"""
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn

from .lle import KerrRing


class RingCascade(nn.Module):
    """A chain of KerrRing layers. Only the first ring receives the signal.

    forward(u) returns the drop-port comb-line powers of every ring, shape
    [batch, T, n_rings, n_modes] (pump line centred), and the list of final fields.
    """

    def __init__(self, rings: Sequence[KerrRing], gains_db: Optional[Sequence[float]] = None,
                 trainable_gain: bool = False):
        super().__init__()
        first = rings[0]
        for ring in rings[1:]:
            if (ring.n_modes, ring.t_r, ring.h) != (first.n_modes, first.t_r, first.h):
                raise ValueError("all rings need the same n_modes, free spectral range and step size")
        self.rings = nn.ModuleList(rings)
        gains = torch.tensor([0.0] * (len(rings) - 1) if gains_db is None else list(gains_db), dtype=first.rdtype)
        if len(gains) != len(rings) - 1:
            raise ValueError("need one gain per link between rings")
        if trainable_gain:
            self._gain_db = nn.Parameter(gains)
        else:
            self.register_buffer("_gain_db", gains)

    def link(self, k: int) -> torch.Tensor:
        """Field coupling from inside ring k-1 to the drive term of ring k."""
        gain = 10.0 ** (self._gain_db[k - 1] / 10)
        return torch.sqrt(self.rings[k].theta1 * gain * self.rings[k - 1].theta2)

    def init_state(self, batch: int = 1, noise: float = 1e-6, seed: Optional[int] = 0) -> List[torch.Tensor]:
        return [ring.init_state(batch, noise, None if seed is None else seed + 1000 * k)
                for k, ring in enumerate(self.rings)]

    def forward(self, u: torch.Tensor, state: Optional[List[torch.Tensor]] = None,
                tbptt: Optional[int] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if u.dim() == 1:
            u = u.unsqueeze(0)
        fields = self.init_state(u.shape[0]) if state is None else list(state)
        first = self.rings[0]
        halves = [torch.exp(ring.linear_operator() * (ring.h / 2)) for ring in self.rings]
        links = [None] + [self.link(k) for k in range(1, len(self.rings))]
        out = []
        for t in range(u.shape[1]):
            if tbptt and t % tbptt == 0:
                fields = [e.detach() for e in fields]
            drive = first.drive(u[:, t])
            for _ in range(first.steps_per_symbol):
                previous = fields[0]                       # field of the ring upstream, before this step
                fields[0] = first.step(fields[0], drive, halves[0])
                for k in range(1, len(self.rings)):
                    current = fields[k]
                    fields[k] = self.rings[k].step(current, links[k] * previous, halves[k])
                    previous = current
            out.append(torch.stack([ring.drop_spectrum(e) for ring, e in zip(self.rings, fields)], dim=1))
        return torch.stack(out, dim=1), fields

    def describe(self) -> dict:
        info = {f"ring{k}": ring.describe() for k, ring in enumerate(self.rings)}
        info["gains_db"] = [float(g) for g in self._gain_db]
        return info


def follower_ring(template: KerrRing, **kw) -> KerrRing:
    """A ring for positions 1.. of a cascade: same grid and material as ``template``, no input lines.

    Its own pump settings are irrelevant, since it is driven only by the ring before it.
    """
    cfg = template.config
    return KerrRing(n_modes=cfg["n_modes"], fsr=cfg["fsr"], wavelength=cfg["wavelength"], n_group=cfg["n_group"],
                    gamma=cfg["gamma"], symbol_time=cfg["symbol_time"], steps_per_symbol=cfg["steps_per_symbol"],
                    **kw)


__all__ = ["RingCascade", "follower_ring"]
