"""Differentiable Lugiato-Lefever simulator of a Kerr microresonator, as a recurrent layer.

Model (Coen et al. 2013 notation, slow time measured in round trips n = t / t_R):

    dE/dn = [-alpha - i*delta0 + i*(L*beta2/2)*omega^2 + i*d_mu] E
            + i*gamma*L*|E|^2 E + sqrt(theta1) * E_in

E(tau) is the intracavity field on a fast-time grid of ``n_modes`` points
(|E|^2 in watts). Its discrete Fourier transform gives the comb-line amplitudes.
Integration uses the fixed-step fourth-order Runge-Kutta interaction-picture
scheme (RK4IP, Hult 2007), built only from FFTs and pointwise operations so
autograd can differentiate through it with respect to every physical parameter.

Ways of feeding a symbol u in [0, 1] into the ring (they can be combined):
  * pump-power encoding (the scheme of Shishavan et al. 2025):
        P_pump = p_avg * (1 + depth * (u - 0.5))
  * pump-detuning encoding (Cuevas et al. 2025): the detuning is shifted by
        detuning_depth * (u - 0.5)   cavity half-linewidths
  * pump-phase encoding (Arabieh et al. 2026): the whole input field is multiplied by
        exp(i * phase_depth * (u - 0.5))
  * optional multi-mode input: extra input lines at the resonances
    ``input_modes`` with complex amplitudes  bias_k + weight_k * (u - 0.5)
    (units sqrt(W)). Physically this is an input comb whose lines are modulated
    by the signal. Because these lines seed the sidebands coherently, the comb
    no longer has to grow out of noise.

Two optional sources of memory beyond the photon lifetime (both off by default):
  * delayed input: the input lines also carry a delayed copy u2 of the signal,
        bias_k + weight_k * (u - 0.5) + delay_weight_k * (u2 - 0.5),
    with u2[t] = u[t - d] passed to ``forward`` (an optical or electronic delay line
    in front of a second modulator); ``set_delay_input`` sets delay_weight;
  * optical feedback loop: the drop-port output is routed back into the input bus
    through a fibre loop whose delay is one symbol, with field transmission
    ``feedback`` (complex: sqrt of the power transmission times the loop phase).
    The light that left the drop port at a given step of the previous symbol
    re-enters at the same step of the current one.

Noise (all off by default, switched on with ``set_noise``):
  * ``rin``       relative rms intensity noise of the source laser, one sample per
                  symbol, common to the pump and all input lines;
  * ``detuning``  rms jitter of the pump-resonance detuning, in units of the cavity
                  half-linewidth, with correlation time ``detuning_corr`` symbols
                  (1 = white, large = slow drift);
  * ``field``     additive white noise on the intracavity field, as rms amplitude
                  gained by each mode per round trip (sqrt(W));
                  ``quantum_field_noise()`` gives the vacuum-fluctuation level.
"""
import math
from typing import Iterable, Optional, Sequence, Tuple

import torch
from torch import nn

C_LIGHT = 299_792_458.0
PLANCK = 6.626_070_15e-34

# Internal scales so that every trainable parameter is O(1).
_SCALES = {"delta0": 1e-3, "beta2": 1e-27, "dint": 1e-3}
TRAINABLE = (
    "delta0", "beta2", "log_q_int", "log_q_ext1", "log_q_ext2", "dint",
    "p_avg", "p_swing", "input_bias", "input_weight", "detuning_depth", "phase_depth",
)
# Raw parameter behind each public name.
_RAW = {"p_avg": "log_p_avg", "p_swing": "mod_depth"}


def _col(x: torch.Tensor) -> torch.Tensor:
    """A per-ring quantity as a column that broadcasts against [batch, n_modes].

    In an ordinary ring it is a scalar and is returned as it is; in a stacked ring
    (``stack_rings``) it has one entry per batch row.
    """
    return x.unsqueeze(-1) if x.dim() == 1 else x


class KerrRing(nn.Module):
    """A Kerr microresonator driven by a symbol sequence.

    Input:  u of shape [batch, T], values in [0, 1] (rescaled symbols).
    Output: drop-port comb-line powers, shape [batch, T, n_modes] (watts,
            fftshifted so the pump line sits at index n_modes // 2),
            sampled at the end of each symbol.

    Any name in ``TRAINABLE`` listed in ``trainable`` becomes an nn.Parameter.
    ``dint`` is a per-mode phase offset (rad / round trip) on top of the
    quadratic dispersion, i.e. a free dispersion profile; it starts at zero.
    """

    def __init__(
        self,
        n_modes: int = 512,
        fsr: float = 200e9,            # Hz
        wavelength: float = 1550e-9,   # m    (not stated in the paper: assumption)
        n_group: float = 1.45,         #      (not stated in the paper: silica assumed)
        gamma: float = 1.6e-2,         # 1/(W m)
        beta2: float = -4.0e-27,       # s^2/m  (= -4e-9 ns^2/m)
        delta0: float = 1.6e-3,        # rad
        q_int: float = 1.0e8,          # split of the total Q = 1e7 is an assumption
        q_ext1: float = 2.2222e7,
        q_ext2: float = 2.2222e7,
        p_avg: float = 0.325,          # W
        p_swing: float = 0.05,         # W, full excursion of the pump power
        detuning_depth: float = 0.0,   # half-linewidths, full excursion of the detuning
        phase_depth: float = 0.0,      # rad, full excursion of the input phase
        input_modes: Sequence[int] = (),   # mode numbers (relative to the pump) of extra input lines
        input_init: float = 0.05,      # sqrt(W), std of the random initial bias / weight of those lines
        symbol_time: float = 1e-9,     # s  (1 GSa/s)
        steps_per_symbol: int = 40,
        trainable: Iterable[str] = (),
        seed: int = 0,
        dtype: torch.dtype = torch.complex128,
    ):
        super().__init__()
        trainable = tuple(trainable)
        unknown = set(trainable) - set(TRAINABLE)
        if unknown:
            raise ValueError(f"unknown trainable parameters: {sorted(unknown)}")
        self.config = {
            "n_modes": n_modes, "fsr": fsr, "wavelength": wavelength, "n_group": n_group,
            "gamma": gamma, "input_modes": list(input_modes), "symbol_time": symbol_time,
            "steps_per_symbol": steps_per_symbol, "trainable": list(trainable),
        }

        self.n_modes = n_modes
        self.cdtype = dtype
        self.rdtype = torch.float64 if dtype == torch.complex128 else torch.float32
        self.t_r = 1.0 / fsr
        self.length = C_LIGHT / (n_group * fsr)
        self.omega0 = 2 * math.pi * C_LIGHT / wavelength
        self.gamma_l = gamma * self.length
        self.steps_per_symbol = steps_per_symbol
        self.h = symbol_time / self.t_r / steps_per_symbol  # round trips per RK step
        self.noise = {"rin": 0.0, "detuning": 0.0, "detuning_corr": 1.0, "field": 0.0}

        mu = torch.fft.fftfreq(n_modes, d=1.0 / n_modes)       # mode index, FFT order
        self.register_buffer("omega_sq", (2 * math.pi * fsr * mu).to(self.rdtype) ** 2)

        init = {
            "delta0": delta0 / _SCALES["delta0"],
            "beta2": beta2 / _SCALES["beta2"],
            "log_q_int": math.log10(q_int),
            "log_q_ext1": math.log10(q_ext1),
            "log_q_ext2": math.log10(q_ext2),
            "log_p_avg": math.log10(p_avg),
            "mod_depth": p_swing / p_avg,
            "detuning_depth": detuning_depth,
            "phase_depth": phase_depth,
        }
        for name, value in init.items():
            public = next((k for k, v in _RAW.items() if v == name), name)
            self._register(name, torch.tensor(value, dtype=self.rdtype), public in trainable)
        self._register("dint", torch.zeros(n_modes, dtype=self.rdtype), "dint" in trainable)
        # Optical feedback loop: complex field transmission as a real pair (0 = no loop).
        self.register_buffer("_feedback", torch.zeros(2, dtype=self.rdtype))
        self.last_loop = None      # light in the feedback loop after the last ``forward`` call

        # Multi-mode input: complex amplitudes stored as [K, 2] real tensors.
        self.input_modes = tuple(int(m) for m in input_modes)
        if 0 in self.input_modes or any(abs(m) >= n_modes // 2 for m in self.input_modes):
            raise ValueError("input_modes must be non-zero and inside the simulated band")
        k = len(self.input_modes)
        gen = torch.Generator().manual_seed(seed)
        for name in ("input_bias", "input_weight"):
            value = input_init * torch.randn(k, 2, generator=gen, dtype=self.rdtype)
            self._register(name, value, name in trainable)
        self.register_buffer("_delay_weight", torch.zeros(k, 2, dtype=self.rdtype))
        tau = torch.arange(n_modes, dtype=self.rdtype) / n_modes
        modes = torch.tensor(self.input_modes, dtype=self.rdtype).unsqueeze(-1)
        self.register_buffer("input_basis", torch.exp(2j * math.pi * modes * tau).to(dtype))  # [K, n_modes]

    def _register(self, name: str, value: torch.Tensor, train: bool) -> None:
        if train:
            self.register_parameter("_" + name, nn.Parameter(value))
        else:
            self.register_buffer("_" + name, value)

    # ---- physical quantities derived from the (possibly trainable) raw parameters ----
    @property
    def delta0(self) -> torch.Tensor:
        return self._delta0 * _SCALES["delta0"]

    @property
    def beta2(self) -> torch.Tensor:
        return self._beta2 * _SCALES["beta2"]

    @property
    def dint(self) -> torch.Tensor:
        return self._dint * _SCALES["dint"]

    @property
    def p_avg(self) -> torch.Tensor:
        return 10.0 ** self._log_p_avg

    @property
    def p_swing(self) -> torch.Tensor:
        return self.p_avg * self._mod_depth

    @property
    def input_bias(self) -> torch.Tensor:
        return torch.view_as_complex(self._input_bias)

    @property
    def input_weight(self) -> torch.Tensor:
        return torch.view_as_complex(self._input_weight)

    def _theta(self, log_q: torch.Tensor) -> torch.Tensor:
        """Power coupling / loss per round trip associated with one Q contribution."""
        return self.t_r * self.omega0 / 10.0 ** log_q

    @property
    def theta1(self) -> torch.Tensor:
        return self._theta(self._log_q_ext1)

    @property
    def theta2(self) -> torch.Tensor:
        return self._theta(self._log_q_ext2)

    @property
    def alpha(self) -> torch.Tensor:
        """Half of the total fractional power loss per round trip."""
        return 0.5 * (self._theta(self._log_q_int) + self.theta1 + self.theta2)

    @property
    def q_total(self) -> torch.Tensor:
        return self.t_r * self.omega0 / (2 * self.alpha)

    def pump_power(self, u: torch.Tensor) -> torch.Tensor:
        """Map symbols in [0, 1] to pump power in watts."""
        return self.p_avg * (1 + self._mod_depth * (u - 0.5))

    def set_input(self, bias: torch.Tensor, weight: torch.Tensor) -> "KerrRing":
        """Set the complex amplitudes (sqrt(W)) of the extra input lines. Returns self."""
        with torch.no_grad():
            self._input_bias.copy_(torch.view_as_real(bias.to(self.cdtype)))
            self._input_weight.copy_(torch.view_as_real(weight.to(self.cdtype)))
        return self

    def set_delay_input(self, weight: torch.Tensor) -> "KerrRing":
        """Complex modulation weights (sqrt(W)) of the delayed signal on the extra input lines. Returns self."""
        with torch.no_grad():
            self._delay_weight.copy_(torch.view_as_real(weight.to(self.cdtype)))
        return self

    def set_feedback(self, transmission: complex) -> "KerrRing":
        """Field transmission of the drop-port-to-input feedback loop (0 switches it off). Returns self."""
        with torch.no_grad():
            self._feedback.copy_(torch.tensor([complex(transmission).real, complex(transmission).imag],
                                              dtype=self.rdtype))
        return self

    @property
    def delay_weight(self) -> torch.Tensor:
        return torch.view_as_complex(self._delay_weight)

    @property
    def feedback(self) -> torch.Tensor:
        return torch.view_as_complex(self._feedback)

    def mean_input_power(self) -> torch.Tensor:
        """Average optical power sent to the ring by the sources (pump + extra input lines), in watts.

        Light returned by a feedback loop is not counted: it is the ring's own output."""
        side = ((self.input_bias.abs() ** 2).sum(-1) + (self.input_weight.abs() ** 2).sum(-1) / 12
                + (self.delay_weight.abs() ** 2).sum(-1) / 12)
        return self.p_avg + side

    def linear_operator(self) -> torch.Tensor:
        """Per-round-trip linear operator in the mode basis (FFT order)."""
        phase = -_col(self.delta0) + 0.5 * self.length * _col(self.beta2) * self.omega_sq + self.dint
        return torch.complex(-_col(self.alpha).expand_as(phase), phase)

    def describe(self) -> dict:
        """Physical parameters as plain numbers, for logging."""
        return {
            "delta0": self.delta0.item(), "beta2": self.beta2.item(),
            "q_int": 10 ** self._log_q_int.item(), "q_ext1": 10 ** self._log_q_ext1.item(),
            "q_ext2": 10 ** self._log_q_ext2.item(), "q_total": self.q_total.item(),
            "p_avg": self.p_avg.item(), "p_swing": self.p_swing.item(),
            "mean_input_power": self.mean_input_power().item(),
            "photon_lifetime_ns": (self.q_total / self.omega0).item() * 1e9,
        }

    # ---- noise ----
    def set_noise(self, rin: float = 0.0, detuning: float = 0.0, detuning_corr: float = 1.0,
                  field: float = 0.0) -> "KerrRing":
        """Switch on noise sources (see the module docstring). Returns self."""
        self.noise = {"rin": rin, "detuning": detuning, "detuning_corr": detuning_corr, "field": field}
        return self

    def quantum_field_noise(self):
        """Vacuum fluctuations entering through all loss channels (half a photon per mode).

        A float, or one value per batch row for a stacked ring.
        """
        photon_power = PLANCK * self.omega0 / (2 * math.pi) / self.t_r      # one photon per round trip
        if self.alpha.dim():
            return torch.sqrt(2 * self.alpha * photon_power / 2)
        return math.sqrt(2 * self.alpha.item() * photon_power / 2)

    def _randn(self, shape, generator) -> torch.Tensor:
        """Standard normal numbers on the ring's device.

        They are drawn on the CPU and moved when ``generator`` is a CPU generator, so a
        seeded run sees the same noise on every device; a generator on the ring's own
        device (or none, off the CPU) draws there, which is what large batches need.
        """
        device = self._dint.device
        if device.type != "cpu" and (generator is None or generator.device.type != "cpu"):
            return torch.randn(shape, generator=generator, dtype=self.rdtype, device=device)
        return torch.randn(shape, generator=generator, dtype=self.rdtype).to(device)

    # ---- driving field ----
    def cw_drive(self, p_in: torch.Tensor) -> torch.Tensor:
        """Coupled-in field of a plain CW pump of power ``p_in`` ([batch]) -> [batch, 1]."""
        return torch.sqrt(self.theta1 * p_in).to(self.cdtype).unsqueeze(-1)

    def drive(self, u: torch.Tensor, u2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Coupled-in field for symbols ``u`` ([batch]) -> [batch, 1] or [batch, n_modes].

        ``u2`` ([batch]): the delayed signal, carried by the input lines with ``delay_weight``."""
        u = u.to(self.rdtype)
        d = self.cw_drive(self.pump_power(u))
        if self.input_modes:
            amp = self.input_bias + self.input_weight * (u.unsqueeze(-1) - 0.5)     # [batch, K]
            if u2 is not None:
                amp = amp + self.delay_weight * (u2.to(self.rdtype).unsqueeze(-1) - 0.5)
            d = d + _col(torch.sqrt(self.theta1)) * (amp.to(self.cdtype) @ self.input_basis)
        if self._encodes("phase_depth"):
            d = d * torch.exp(1j * _col(self._phase_depth) * (u.unsqueeze(-1) - 0.5))
        return d

    def _encodes(self, name: str) -> bool:
        value = getattr(self, "_" + name)
        return value.requires_grad or bool((value != 0).any())

    # ---- integration ----
    def init_state(self, batch: int = 1, noise: float = 1e-6, seed: Optional[int] = 0) -> torch.Tensor:
        """Empty cavity plus weak complex noise (each batch row gets different noise)."""
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        re = torch.randn(batch, self.n_modes, generator=gen, dtype=self.rdtype)
        im = torch.randn(batch, self.n_modes, generator=gen, dtype=self.rdtype)
        return (noise * torch.complex(re, im)).to(self._dint.device)

    def soliton_state(self, batch: int = 1, noise: float = 1e-6, seed: Optional[int] = 0) -> torch.Tensor:
        """A single dissipative Kerr soliton on the lower-branch CW background (plus seed noise).

        This is the approximate analytic soliton of the LLE; it relaxes to the exact one
        within a few photon lifetimes if the operating point supports solitons, and
        decays to the CW state otherwise. Needs anomalous dispersion and delta0 > 0.
        """
        import numpy as np
        a, d, gl = self.alpha.item(), self.delta0.item(), self.gamma_l
        drive = math.sqrt(self.theta1.item() * self.p_avg.item())
        roots = np.roots([gl ** 2, -2 * d * gl, a ** 2 + d ** 2, -drive ** 2])
        p_cw = min(r.real for r in roots if abs(r.imag) < 1e-9 * abs(r) and r.real > 0)
        background = drive / complex(a, d - gl * p_cw)
        peak = math.sqrt(2 * d / gl)
        width = math.sqrt(self.length * abs(self.beta2.item()) / (2 * d))        # seconds
        f = math.sqrt(gl * drive ** 2 / a ** 3)
        phase = math.acos(min(1.0, math.sqrt(8 * d / a) / (math.pi * f)))
        tau = (torch.arange(self.n_modes, dtype=self.rdtype) / self.n_modes - 0.5) * self.t_r
        pulse = peak / torch.cosh(tau / width) * complex(math.cos(phase), math.sin(phase))
        soliton = (background + pulse).to(self.cdtype).unsqueeze(0).to(self._dint.device)
        return soliton + self.init_state(batch, noise, seed)

    def step(self, e: torch.Tensor, drive: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
        """One RK4IP step of ``self.h`` round trips."""
        h, gl = self.h, self.gamma_l

        def nonlin(a):
            return h * (1j * gl * (a.real ** 2 + a.imag ** 2) * a + drive)

        def lin(a):
            return torch.fft.ifft(half * torch.fft.fft(a))

        e_ip = lin(e)
        k1 = lin(nonlin(e))
        k2 = nonlin(e_ip + k1 / 2)
        k3 = nonlin(e_ip + k2 / 2)
        k4 = nonlin(lin(e_ip + k3))
        return lin(e_ip + k1 / 6 + k2 / 3 + k3 / 3) + k4 / 6

    def evolve(
        self,
        e: torch.Tensor,
        drive: torch.Tensor,
        n_steps: Optional[int] = None,
        detuning_offset: Optional[torch.Tensor] = None,
        field_noise: float = 0.0,
        generator: Optional[torch.Generator] = None,
        extra: Optional[torch.Tensor] = None,
        record: Optional[list] = None,
    ) -> torch.Tensor:
        """Evolve the field under a constant ``drive`` (from ``drive`` / ``cw_drive``) for one symbol.

        ``detuning_offset`` ([batch], rad) shifts the detuning of each batch row;
        ``field_noise`` adds white noise after every step (see ``set_noise``); it may be a
        tensor with one level per batch row. ``extra`` ([steps, batch, n_modes]) is added to the
        drive step by step (light from a feedback loop); ``record``, a list, receives the field
        at the start of every step.
        """
        lop = self.linear_operator()
        if detuning_offset is not None:
            lop = (lop if lop.dim() == 2 else lop.unsqueeze(0)) - 1j * detuning_offset.unsqueeze(-1)
        half = torch.exp(lop * (self.h / 2))
        noisy = torch.is_tensor(field_noise) or bool(field_noise)
        if torch.is_tensor(field_noise):
            field_noise = _col(field_noise)
        sigma = field_noise * math.sqrt(self.h * self.n_modes / 2)     # per grid point and quadrature
        for s in range(n_steps or self.steps_per_symbol):
            if record is not None:
                record.append(e)
            e = self.step(e, drive if extra is None else drive + extra[s], half)
            if noisy:
                e = e + sigma * torch.complex(self._randn(e.shape, generator), self._randn(e.shape, generator))
        return e

    def drop_spectrum(self, e: torch.Tensor) -> torch.Tensor:
        """Power in each comb line at the drop port (watts), pump line centred."""
        a = torch.fft.fft(e) / self.n_modes
        return torch.fft.fftshift(_col(self.theta2) * (a.real ** 2 + a.imag ** 2), dim=-1)

    def forward(
        self,
        u: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        tbptt: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        u2: Optional[torch.Tensor] = None,
        loop: Optional[torch.Tensor] = None,
        rin: Optional[torch.Tensor] = None,
        detuning: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the ring over a symbol sequence.

        ``tbptt``: if set, the field is detached every ``tbptt`` symbols, so gradients
        flow through at most that many symbols of history (truncated BPTT). This is
        what keeps gradients finite when the ring operates in a chaotic regime.
        ``generator`` seeds the noise sources enabled with ``set_noise``; every batch
        row gets independent noise.
        ``u2`` ([batch, T]): delayed signal for the delayed-input lines (see the module docstring).
        ``loop``: light in the feedback loop at the start, [steps, batch, n_modes] intracavity fields
        of the previous symbol (default: an empty loop); continue a sequence fed in blocks by passing
        ``last_loop`` of the previous call, which every call with an active loop sets.
        ``rin`` ([batch, T], standard normal): the intensity-noise samples to use instead of drawing them,
        e.g. the same samples for rings fed from one source; the other noise sources stay independent.
        ``detuning`` ([batch, T], unit variance): the detuning-jitter process itself (the per-symbol drift that
        is scaled by the jitter amplitude) to use instead of drawing it, e.g. laser frequency noise shared by
        several rings, or a known sequence a readout is given as a reference.
        """
        if u.dim() == 1:
            u = u.unsqueeze(0)
        if u2 is not None and u2.dim() == 1:
            u2 = u2.unsqueeze(0)
        e = self.init_state(u.shape[0]) if state is None else state
        looped = bool((self._feedback != 0).any())
        if looped:
            # drop-port field sqrt(theta2) E, coupled back in through the input port: sqrt(theta1)
            coupling = (_col(torch.sqrt(self.theta1 * self.theta2)) * _col(self.feedback)).to(self.cdtype)
            if loop is None:
                loop = torch.zeros((self.steps_per_symbol,) + tuple(e.shape), dtype=self.cdtype, device=e.device)
        noise, batch = self.noise, u.shape[0]
        if noise["detuning"] and detuning is None:
            rho = math.exp(-1.0 / noise["detuning_corr"]) if noise["detuning_corr"] > 0 else 0.0
            drift = self._randn((batch,), generator)       # stationary, unit variance
        out = []
        for t in range(u.shape[1]):
            if tbptt and t % tbptt == 0:
                e = e.detach()
            drive, offset = self.drive(u[:, t], None if u2 is None else u2[:, t]), None
            if self._encodes("detuning_depth"):
                offset = self._detuning_depth * self.alpha * (u[:, t].to(self.rdtype) - 0.5)
            if noise["rin"]:
                sample = (rin[:, t: t + 1].to(device=drive.device, dtype=self.rdtype) if rin is not None
                          else self._randn((batch, 1), generator))
                drive = drive * (1 + 0.5 * noise["rin"] * sample)
            if noise["detuning"]:
                if detuning is not None:
                    drift = detuning[:, t].to(device=drive.device, dtype=self.rdtype)
                else:
                    drift = rho * drift + math.sqrt(1 - rho ** 2) * self._randn((batch,), generator)
                jitter = noise["detuning"] * self.alpha * drift
                offset = jitter if offset is None else offset + jitter
            if looped:
                fields = []
                e = self.evolve(e, drive, detuning_offset=offset, field_noise=noise["field"], generator=generator,
                                extra=coupling * loop, record=fields)
                loop = torch.stack(fields)
            else:
                e = self.evolve(e, drive, detuning_offset=offset, field_noise=noise["field"], generator=generator)
            out.append(self.drop_spectrum(e))
        self.last_loop = loop if looped else None
        return torch.stack(out, dim=1), e

    # ---- analytic references, used for validation ----
    def cw_input_power(self, p_cav: torch.Tensor) -> torch.Tensor:
        """Pump power that sustains a homogeneous intracavity power ``p_cav`` (steady state)."""
        a, d, gl = self.alpha, self.delta0, self.gamma_l
        return p_cav * (a ** 2 + (d - gl * p_cav) ** 2) / self.theta1

    def mi_gain(self, p_cav: torch.Tensor) -> torch.Tensor:
        """Modulation-instability power gain per round trip of each mode (fftshifted)."""
        gl = self.gamma_l
        phase = -self.delta0 + 0.5 * self.length * self.beta2 * self.omega_sq + self.dint + 2 * gl * p_cav
        disc = torch.complex((gl * p_cav) ** 2 - phase ** 2, torch.zeros_like(phase))
        return torch.fft.fftshift(2 * (-self.alpha + torch.sqrt(disc).real))


def stack_rings(rings: Sequence[KerrRing], repeats: int = 1) -> KerrRing:
    """One ring object that simulates several rings side by side, one per batch row.

    Batch row i of the result uses the parameters of ``rings[i // repeats]``, so the batch
    passed to ``forward`` must have ``len(rings) * repeats`` rows. The rings must share the
    grid, the material and the set of input lines; everything else (Q factors, detuning,
    dispersion, powers, input amplitudes, encoding depths) may differ. The result is for
    simulation only: it has no trainable parameters, and ``describe``, ``soliton_state``
    and the analytic references work on the individual rings, not on the stack.
    """
    shared = {k: v for k, v in rings[0].config.items() if k != "trainable"}
    for ring in rings:
        if {k: v for k, v in ring.config.items() if k != "trainable"} != shared:
            raise ValueError("stacked rings must share the grid, the material and the input lines")
    out = KerrRing(**shared, dtype=rings[0].cdtype)
    for name in list(out._buffers):
        if name.startswith("_"):
            value = torch.stack([getattr(ring, name).detach() for ring in rings]).repeat_interleave(repeats, 0)
            out.register_buffer(name, value)
    return out.to(rings[0]._dint.device)


def save_ring(ring: KerrRing, path: str) -> None:
    torch.save({"config": ring.config, "state": ring.state_dict()}, path)


def load_ring(path: str, **overrides) -> KerrRing:
    """Rebuild a ring saved by ``save_ring``.

    ``overrides`` replace constructor settings, e.g. ``steps_per_symbol`` or
    ``n_modes`` (to re-run trained parameters on a finer grid). Grid-sized buffers
    are then rebuilt rather than loaded, and a trained per-mode ``dint`` profile is
    copied mode by mode (modes outside the original grid get zero).
    """
    blob = torch.load(path, weights_only=True)
    ring = KerrRing(**{**blob["config"], **overrides})
    own = ring.state_dict()
    state = {k: v for k, v in blob["state"].items() if v.shape == own[k].shape}
    if "_dint" not in state:
        old = blob["state"]["_dint"]
        n_old, n_new = len(old), ring.n_modes
        if n_new < n_old:
            raise ValueError("cannot load a ring onto a smaller spectral grid")
        mu = torch.fft.fftfreq(n_old, d=1.0 / n_old).long()          # mode numbers, FFT order
        new = torch.zeros(n_new, dtype=old.dtype)
        new[mu % n_new] = old
        state["_dint"] = new
    ring.load_state_dict(state, strict=False)
    return ring
