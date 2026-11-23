"""Figures of the paper, from the result files in results/ (no new simulations except figure 3).

    python experiments/figures.py            # all figures into figures/
    python experiments/figures.py 4 8        # only these

Style: one validated categorical palette (blue, orange, aqua; checked with the data-viz validator for
colour-blind separation, all pairs), thin marks, hairline grid, one y-axis per panel, legends whenever
two or more series share a panel. Sizes follow journal columns: 3.4 in single, 7.0 in double.
"""
import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SINGLE, DOUBLE = 3.4, 7.0

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 150,
})

GROUP_LABEL = {"power": "power", "detuning": "detuning", "phase": "phase", "power_soliton": "power\n(soliton)",
               "detuning_soliton": "detuning\n(soliton)", "phase_soliton": "phase\n(soliton)",
               "comb_uniform": "uniform\ncomb", "comb_mask": "spectral\nphase mask"}
ORDER = list(GROUP_LABEL)


def family_color(group):
    """Encoding family: CW single pump (blue), soliton single pump (orange), input comb (aqua)."""
    return AQUA if group.startswith("comb") else ORANGE if group.endswith("soliton") else BLUE


def read(name):
    with open(os.path.join(RES, name), newline="") as f:
        return list(csv.DictReader(f))


def gmean(values):
    values = [float(v) for v in values]
    return math.exp(sum(math.log(v) for v in values) / len(values))


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def panel_tag(ax, tag):
    ax.text(-0.02, 1.04, tag, transform=ax.transAxes, fontweight="bold", fontsize=9, ha="right", va="bottom")


def ref_line(ax, y, text, style, x=None, va=None):
    """A reference level (a baseline) as a gray line; it is named in the legend, not on the plot."""
    return ax.axhline(y, color=INK2, linestyle=style, linewidth=0.9, zorder=1, label=text)


# ---- figure 1: scheme -------------------------------------------------------------------------

def fig1():
    """Schematic of the reservoir: input comb split into a static and a modulated branch, each through a
    spectral phase shaper (the phase mask), recombined into an add-drop Kerr ring; drop-port lines are
    demultiplexed onto a photodiode array and combined by a linear readout with delay taps."""
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
    HA, HB = 2.55, 1.85                           # heights of panel a (scheme) and panel b (model), inches
    H = HA + HB
    fig = plt.figure(figsize=(DOUBLE, H))
    ax = fig.add_axes([0, HB / H, 1, HA / H])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.1)
    ax.axis("off")
    ax.text(0.05, 5.0, "a", fontweight="bold", fontsize=9, ha="left", va="top")
    OPT, ELEC = BLUE, ORANGE                      # optical paths, electrical signals

    def box(x, y, w, h, text, sub=None):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor="#f4f3f0", edgecolor=INK2, linewidth=0.8, zorder=3))
        ax.text(x, y + (0.13 if sub else 0), text, ha="center", va="center", fontsize=6.6, zorder=4)
        if sub:
            ax.text(x, y - 0.2, sub, ha="center", va="center", fontsize=5.6, color=INK2, zorder=4)

    def wire(points, color=OPT, arrow=True, lw=1.3):
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            last = (x1, y1) == points[-1]
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>" if (arrow and last) else "-",
                                         mutation_scale=7, color=color, linewidth=lw, shrinkA=0, shrinkB=0,
                                         zorder=2))

    # source and comb
    box(0.85, 2.6, 1.3, 0.75, "CW laser", "1550 nm")
    box(2.75, 2.6, 1.45, 0.75, "200 GHz comb", "FSR-locked")
    wire([(1.5, 2.6), (2.03, 2.6)])
    # split into two branches
    wire([(3.48, 2.6), (3.85, 2.6), (3.85, 3.65), (4.6, 3.65)], arrow=False)
    wire([(3.85, 2.6), (3.85, 1.55), (4.6, 1.55)])
    box(5.4, 1.55, 1.55, 0.72, "modulator", "one MZM")
    box(7.2, 3.65, 1.6, 0.72, "phase shaper", r"static: $b_\mu$")
    box(7.2, 1.55, 1.6, 0.72, "phase shaper", r"signal: $w_\mu$")
    wire([(4.6, 3.65), (6.4, 3.65)])
    wire([(6.18, 1.55), (6.4, 1.55)])
    # signal into the modulator
    ax.text(5.4, 0.55, "input u(t)", ha="center", va="center", fontsize=6.6, color=INK)
    wire([(5.4, 0.72), (5.4, 1.19)], color=ELEC, lw=1.1)
    # recombine
    wire([(8.0, 3.65), (8.55, 3.65), (8.55, 2.75)])
    wire([(8.0, 1.55), (8.55, 1.55), (8.55, 2.45)])
    ax.add_patch(Circle((8.55, 2.6), 0.13, facecolor="white", edgecolor=INK2, linewidth=0.8, zorder=3))
    ax.text(8.55, 2.95 + 0.9, "", fontsize=6)
    # ring: input bus (top), drop bus (bottom)
    rx, ry, rr = 10.0, 2.6, 0.55
    wire([(8.68, 2.6), (9.0, 2.6), (9.0, 3.25), (11.2, 3.25)], arrow=True)
    ax.text(11.25, 3.25, "through", ha="left", va="center", fontsize=5.8, color=INK2)
    ax.add_patch(Circle((rx, ry), rr, facecolor="none", edgecolor=OPT, linewidth=2.2, zorder=3))
    ax.text(rx, ry, "Kerr\nring", ha="center", va="center", fontsize=6.2, zorder=4)
    ax.text(rx + rr + 0.15, ry, r"$Q_\mathrm{int}$, $\zeta$", ha="left", va="center", fontsize=5.8, color=INK2)
    ax.text(10.0, 3.36, r"$Q_{\mathrm{ext},1}$", ha="center", va="bottom", fontsize=5.8, color=INK2)
    wire([(10.75, 1.95), (9.25, 1.95)], arrow=False)                       # drop bus under the ring
    wire([(10.75, 1.95), (11.45, 1.95), (11.45, 1.38)])
    ax.text(10.0, 1.7, r"drop port, $Q_{\mathrm{ext},2}$", ha="center", va="top", fontsize=5.8, color=INK2)
    # demux + photodiodes + readout
    box(11.45, 1.05, 1.2, 0.62, "demux", "31 lines")
    for k in range(5):
        y = 0.25 - 0.0 + k * 0.0
    box(13.05, 1.05, 1.3, 0.62, "photodiodes", "per line")
    wire([(12.05, 1.05), (12.38, 1.05)])
    box(13.05, 2.6, 1.3, 0.75, "readout", "ridge + taps")
    wire([(13.05, 1.36), (13.05, 2.22)], color=ELEC, lw=1.1)
    ax.text(13.05, 3.4, "output y(t)", ha="center", va="center", fontsize=6.6)
    wire([(13.05, 2.98), (13.05, 3.22)], color=ELEC, lw=1.1)

    # insets: input comb with random phases, drop-port spectrum
    def inset(x0, y0, w, h, heights, phases=None, title=""):
        ia = fig.add_axes([x0 / 14, (HB + y0 / 5.1 * HA) / H, w / 14, h / 5.1 * HA / H])
        n = len(heights)
        xs = np.arange(n) - n // 2
        ia.vlines(xs, 0, heights, color=OPT, linewidth=1.0)
        if phases is not None:
            ia.scatter(xs, np.asarray(heights) + 0.12, s=7, c=phases, cmap="twilight", vmin=-np.pi, vmax=np.pi,
                       linewidths=0, zorder=3)
        ia.set_ylim(0, 1.35)
        ia.set_xticks([])
        ia.set_yticks([])
        for side in ("top", "right", "left"):
            ia.spines[side].set_visible(False)
        ia.grid(False)
        ia.set_title(title, fontsize=5.6, color=INK2, pad=1.5)
        ia.patch.set_alpha(0)
    rng = np.random.default_rng(3)
    amps = np.r_[0.55 * np.ones(8), 1.0, 0.55 * np.ones(8)]
    inset(6.45, 4.25, 1.5, 0.62, amps, rng.uniform(-np.pi, np.pi, 17), "random phase per line")
    drop = np.exp(-np.abs(np.arange(31) - 15) / 6.0) * (0.6 + 0.4 * rng.random(31))
    drop[15] = 1.0
    inset(11.35, 3.75, 1.9, 0.62, drop, None, "drop-port lines")
    # key: optical path vs electrical signal
    for i, (color, text) in enumerate(((OPT, "optical path"), (ELEC, "electrical signal"))):
        y = 0.75 - 0.38 * i
        ax.plot([0.3, 0.85], [y, y], color=color, linewidth=1.3)
        ax.text(0.98, y, text, va="center", fontsize=6, color=INK2)

    # panel b: the simulation model as a signal chain, with the points where each noise enters
    bx = fig.add_axes([0, 0, 1, HB / H])
    bx.set_xlim(0, 14)
    bx.set_ylim(0, 3.7)
    bx.axis("off")
    bx.text(0.05, 3.62, "b", fontweight="bold", fontsize=9, ha="left", va="top")
    stages = [  # (x left, width, title, formula lines, note)
        (0.15, 2.75, "input mask",
         [r"$\tilde E_{\mathrm{in},\mu}=b_\mu+w_\mu\,(u_t-\frac{1}{2})$",
          r"$\tilde E_{\mathrm{in},0}=\sqrt{P_p[1+2d\,(u_t-\frac{1}{2})]}$"], r"$\mu=\pm1..\pm8$, random $b_\mu, w_\mu$"),
        (3.12, 3.85, "LLE (512 modes, RK4IP)",
         [r"$\frac{d\tilde E_\mu}{dn}=\left[-\alpha-i\delta_0+i\frac{L\beta_2}{2}\omega_\mu^2\right]\tilde E_\mu$",
          r"$+\,i\gamma L\,\mathcal{F}_\mu\left[|E|^2E\right]+\sqrt{\theta_1}\,\tilde E_{\mathrm{in},\mu}$"], r"$\zeta=\delta_0/\alpha$, one symbol $T=1$ ns"),
        (7.2, 1.75, "drop port", [r"$P_\mu=\theta_2|\tilde E_\mu|^2$"], r"$\mu=-15..15$"),
        (9.2, 2.25, "detection", [r"$x_\mu=\log_{10}(\hat P_\mu+P_\mathrm{floor})$"], "31 photodiodes"),
        (11.7, 2.15, "linear readout", [r"$\hat y_t=\sum_{k=0}^{7}W_k\,\mathbf{x}_{t-k}$"], "ridge regression"),
    ]
    yc, hh = 2.25, 1.55
    for i, (x0, w, title, lines, note) in enumerate(stages):
        bx.add_patch(FancyBboxPatch((x0, yc - hh / 2), w, hh, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor="#f4f3f0", edgecolor=INK2, linewidth=0.8, zorder=3))
        bx.text(x0 + w / 2, yc + hh / 2 - 0.2, title, ha="center", va="center", fontsize=6.4, zorder=4)
        n = len(lines)
        for j, line in enumerate(lines):
            bx.text(x0 + w / 2, yc + 0.08 - (j - (n - 1) / 2) * 0.42, line, ha="center", va="center",
                    fontsize=6.4, zorder=4)
        bx.text(x0 + w / 2, yc - hh / 2 + 0.17, note, ha="center", va="center", fontsize=5.3, color=INK2, zorder=4)
        if i + 1 < len(stages):
            nx = stages[i + 1][0]
            bx.add_patch(FancyArrowPatch((x0 + w + 0.02, yc), (nx - 0.02, yc), arrowstyle="-|>", mutation_scale=7,
                                         color=INK2, linewidth=1.0, zorder=2))
    noises = [  # (x, text) — dashed arrows up into the stage where each noise enters
        (1.52, "laser intensity noise\n(RIN $10^{-3}$)"),
        (4.15, "detuning jitter\n(laser frequency noise)"),
        (5.85, "intracavity\nvacuum noise"),
        (10.32, "shot noise +\ndetector noise ($0.3\\,\\mu$W)"),
    ]
    for x, text in noises:
        bx.add_patch(FancyArrowPatch((x, 0.72), (x, yc - hh / 2 - 0.02), arrowstyle="-|>", mutation_scale=6,
                                     color=MUTED, linewidth=0.9, linestyle=(0, (2, 1.5)), zorder=2))
        bx.text(x, 0.62, text, ha="center", va="top", fontsize=5.6, color=INK2, linespacing=1.1)
    save(fig, "fig1_scheme")


# ---- figure 2: operating map -------------------------------------------------------------------

def fig2():
    rows = read("operating_map.csv")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.7), sharey=True)
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    cmap = LinearSegmentedColormap.from_list("seq", SEQ[::-1])        # dark = low error = good
    for ax, noise, title, tag in zip(axes, ("none", "realistic"), ("no noise", "realistic noise, PIN detectors"),
                                     "ab"):
        mine = [r for r in rows if r["noise"] == noise]
        qs = sorted({float(r["log_q_ext1"]) for r in mine})
        ps = sorted({float(r["p_total"]) for r in mine})
        grid = np.full((len(ps), len(qs)), np.nan)
        for r in mine:
            v = float(r["test_31x8"])
            grid[ps.index(float(r["p_total"])), qs.index(float(r["log_q_ext1"]))] = min(v, 1.0) if v == v else 1.0
        mesh = ax.pcolormesh(10 ** np.array(qs), np.array(ps) * 1e3, grid, cmap=cmap,
                             norm=LogNorm(vmin=1e-6, vmax=1), shading="nearest")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(False)
        ax.set_xlabel("input coupling $Q_{ext,1}$")
        ax.set_title(title, loc="left")
        panel_tag(ax, tag)
    axes[0].set_ylabel("total input power (mW)")
    bar = fig.colorbar(mesh, ax=axes, shrink=0.9, pad=0.02)
    bar.set_label("test NMSE, Mackey-Glass (lower is better)")
    bar.outline.set_visible(False)
    save(fig, "fig2_operating_map")


# ---- figure 3: chaotic vs coherently seeded: two runs, same input ----------------------------

def fig3():
    sys.path.insert(0, ROOT)
    sys.path.insert(0, HERE)
    import torch
    import compare_encodings as ce
    from combrc import KerrRing, make_task
    from combrc.evaluate import simulate
    from combrc.readout import select_lines
    u = torch.tensor(make_task("mackey_glass_h10", 700, seed=0)[0], dtype=torch.float64)
    chaotic = KerrRing(n_modes=256, steps_per_symbol=20)          # the chaotic-comb parameters of the original scheme
    mask = ce.build(ce.sample("comb_mask", 45), 256, 20)          # the mask chosen in the noise-floor study
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE, 3.8), sharex=True, gridspec_kw={"hspace": 0.45})
    for ax, ring, title, tag in zip(axes, (chaotic, mask), ("chaotic comb (single pump)", "spectral phase mask"),
                                    "ab"):
        spectra = simulate(ring, u, warmup=300, n=300, seeds=(0, 1))
        line = select_lines(spectra, 31)[..., 15 + 3] * 1e3            # third line beside the pump, mW
        t = np.arange(line.shape[1])
        ax.plot(t, line[0], color=BLUE, linewidth=1.0, label="run 1")
        ax.plot(t, line[1], color=ORANGE, linewidth=1.0, linestyle=(0, (3, 1.5)), label="run 2 (other initial noise)")
        c = np.corrcoef(line[0], line[1])[0, 1]
        ax.set_title(f"{title}: correlation of the two runs {c:.2f}", loc="left")
        ax.set_ylabel("line power (mW)")
        panel_tag(ax, tag)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.0))
    fig.subplots_adjust(top=0.86)
    axes[1].set_xlabel("symbol")
    save(fig, "fig3_two_runs")


# ---- figure 4: equal-budget encoding comparison ------------------------------------------------

def fig4():
    tasks = (("robust_noisy_mg_summary.csv", "Mackey-Glass, 10 steps ahead", None),
             ("robust_noisy_ce_summary.csv", "channel equalisation", None),
             ("robust_noisy_narma_summary.csv", "NARMA-10", None))
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.6))
    for ax, (name, title, _), tag in zip(axes, tasks, "abc"):
        rows = {r["group"]: r for r in read(name)}
        groups = [g for g in ORDER if g in rows]
        x = np.arange(len(groups))
        vals = [float(rows[g]["fresh_test"]) for g in groups]
        ax.bar(x, vals, width=0.62, color=[family_color(g) for g in groups], zorder=2)
        any_row = rows[groups[0]]
        esn = ref_line(ax, float(any_row["baseline_esn_fresh"]), "ESN, same size", "--")
        lin = ref_line(ax, float(any_row["baseline_linear"]), "best linear autoregression", ":")
        ax.set_yscale("log")
        low = min(vals + [float(any_row["baseline_esn_fresh"])])
        ax.set_ylim(low / 2.5, max(vals) * 1.6)
        ax.set_xticks(x)
        ax.set_xticklabels([GROUP_LABEL[g].replace("\n", " ") for g in groups], fontsize=6, rotation=90)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
        panel_tag(ax, tag)
    axes[0].set_ylabel("test NMSE, new runs (lower is better)")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=BLUE, label="single pump, CW start"), Patch(color=ORANGE, label="single pump, soliton"),
                        Patch(color=AQUA, label="input comb"), esn, lin], loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save(fig, "fig4_encodings")


# ---- figure 5: every random operating point --------------------------------------------------

def fig5():
    rows = read("robust_noisy_mg_trials.csv")
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.4))
    rng = np.random.default_rng(0)
    for i, g in enumerate(ORDER):
        v = np.array([float(r["test_31x8"]) for r in rows if r["group"] == g])
        v = v[np.isfinite(v)]
        v = np.clip(v, 1e-5, 10)
        ax.scatter(i + rng.uniform(-0.28, 0.28, len(v)), v, s=4, color=family_color(g), alpha=0.35, linewidths=0,
                   zorder=2)
        ax.plot([i - 0.32, i + 0.32], [np.median(v)] * 2, color=INK, linewidth=1.2, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([GROUP_LABEL[g] for g in ORDER], fontsize=6)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("test NMSE, Mackey-Glass\n(realistic noise)")
    ax.set_title("all 256 random operating points per encoding and series (3 series); black: median; "
                 "values above 10 drawn at 10", loc="left")
    save(fig, "fig5_distributions")


# ---- figure 6: noise ------------------------------------------------------------------------

def fig6():
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    row = {r["group"]: r for r in read("floor_breakdown.csv")}["comb_mask"]
    names = [("none", "none"), ("rin", "laser\nRIN"), ("detuning", "detuning\njitter"),
             ("field", "vacuum\nin cavity"), ("shot", "shot\nnoise"), ("optical + shot", "all")]
    vals = [float(row[k]) for k, _ in names]
    ax.bar(range(len(vals)), vals, width=0.6, color=BLUE, zorder=2)
    ax.set_yscale("log")
    ax.set_yticks([1e-5, 2e-5, 5e-5, 1e-4, 2e-4], ["1e-5", "2e-5", "5e-5", "1e-4", "2e-4"])
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([n for _, n in names], fontsize=6)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("test NMSE, Mackey-Glass")
    ax.set_title("one noise source at a time, ideal detectors", loc="left")
    panel_tag(ax, "a")

    ax = axes[1]
    levels = [0.0, 0.001, 0.003, 0.01, 0.03]
    sweep = {}
    import re
    for line in open(os.path.join(RES, "detuning_sweep.log")):
        m = re.match(r"jitter ([\d.]+)%\s+(\S+)\s+([\d.e+-]+)", line)
        if m:
            sweep[(float(m.group(1)) / 100, m.group(2))] = float(m.group(3))
    for name in ("colab_detuning_sweep_0.001.csv", "colab_detuning_sweep_0.003.csv", "colab_detuning_sweep_0.03.csv"):
        for r in read(name):
            sweep[(float(r["detuning_jitter"]), r["group"])] = float(r["test"])
    for r in read("floor_mg_pin_summary.csv"):
        sweep.setdefault((0.01, r["group"]), float(r["fresh_test"]))
    esn = float(read("colab_detuning_sweep_0.03.csv")[0]["esn"])
    x = np.arange(len(levels))
    for g, color, label in (("comb_mask", BLUE, "phase mask"), ("mask_drop", ORANGE, "stronger drop coupling"),
                            ("mask_slow2", AQUA, "2 ns symbols")):
        y = [sweep.get((lv, g), np.nan) for lv in levels]
        ax.plot(x, y, color=color, marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.8,
                label=label, zorder=3)
    ref_line(ax, esn, "ESN, same size", "--")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["0", "0.1", "0.3", "1", "3"])
    ax.minorticks_off()
    ax.set_yticks([5e-5, 1e-4, 2e-4, 5e-4], ["5e-5", "1e-4", "2e-4", "5e-4"])
    ax.set_xlabel("detuning jitter (% of half-linewidth, white)")
    ax.set_ylabel("test NMSE, PIN detectors")
    ax.set_title("readout-side design vs assumed jitter", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    panel_tag(ax, "b")
    fig.tight_layout()
    save(fig, "fig6_noise")


# ---- figure 7: gradient design vs random search ----------------------------------------------

def fig7():
    rows = read("finetune_summary.csv")
    labels = {"random search, 32 draws": "random search, 32 points",
              "random search, 256 draws": "random search, 256 points",
              "32 draws + fine-tuning of the best": "best of 32 + gradient fine-tuning",
              "fine-tuning of middle-ranked draws": "gradient fine-tuning of mid-ranked points"}
    fig, ax = plt.subplots(figsize=(SINGLE, 1.9))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        ax.plot([float(r["min_test"]), float(r["max_test"])], [yi, yi], color=BLUE, linewidth=1.2, zorder=2)
        ax.plot(float(r["test"]), yi, "o", color=BLUE, markersize=5, markeredgecolor="white", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([labels.get(r["method"], r["method"]) for r in rows], fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xticks([2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2], ["2e-4", "5e-4", "1e-3", "2e-3", "5e-3", "1e-2"])
    ax.minorticks_off()
    ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("test NMSE, Mackey-Glass, new noise\n(dot: mean over series, bar: range)")
    save(fig, "fig7_gradient")


# ---- figure S4 (was 8): readout-side design in closed loop ----------------------------------------

def figS4():
    def controllers(name):
        return {r["controller"]: r for r in read(name)}
    fig = plt.figure(figsize=(DOUBLE, 4.9))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.05, 1], hspace=0.55)
    top = outer[0].subgridspec(3, 1, hspace=0.25)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.1, 1], wspace=0.3)

    # a: trajectories of the first trial in a window with level steps and a fast sine
    blob = np.load(os.path.join(RES, "closed_loop_track_readout_traces.npz"))
    names = list(blob["names"])
    goal = blob["goal"]
    window = slice(940, 1100)
    t = np.arange(len(goal))[window]
    shown = (("ESN, 31 nodes", "ESN", INK2), ("ring comb_mask, PIN detectors", "phase mask ring", BLUE),
             ("ring mask_slow2, PIN detectors", "2 ns ring", AQUA))
    for i, (key, label, color) in enumerate(shown):
        ax = fig.add_subplot(top[i])
        x = blob["traces"][names.index(key)]
        ax.plot(t + 1, goal[window], color=MUTED, linewidth=1.0, label="commanded")    # x[t+1] aims at goal[t]
        ax.plot(t, x[window], "o", color=color, markersize=1.8, label=label)
        ax.set_xlim(t[0], t[-1] + 1)
        ax.set_ylim(-1.3, 1.3)
        ax.set_yticks([-1, 0, 1])
        ax.text(1.01, 0.5, label, transform=ax.transAxes, fontsize=6.5, va="center", ha="left", color=INK)
        if i < 2:
            ax.set_xticklabels([])
        if i == 0:
            ax.set_title("Hénon map, first trial: commanded trajectory (gray line) and the controlled state (dots), "
                         "PIN detectors", loc="left")
            panel_tag(ax, "a")
        if i == 1:
            ax.set_ylabel("x")
    ax.set_xlabel("step")

    ax = fig.add_subplot(bottom[0])
    rows = controllers("closed_loop_track_readout.csv")
    items = [("linear predictor", "linear\npredictor", INK2), ("ESN, 31 nodes", "ESN", INK2),
             ("ring comb_mask, PIN detectors", "phase mask", BLUE),
             ("ring mask_drop, PIN detectors", "strong\ndrop port", ORANGE),
             ("ring mask_slow2, PIN detectors", "2 ns\nsymbols", AQUA)]
    vals = [float(rows[k]["tracking_nmse"]) for k, _, _ in items]
    ax.bar(range(len(items)), vals, width=0.6, color=[c for _, _, c in items], zorder=2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([n for _, n, _ in items], fontsize=6)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("tracking NMSE (lower is better)")
    ax.set_title("tracking error over 8 trials", loc="left")
    panel_tag(ax, "b")

    ax = fig.add_subplot(bottom[1])
    white = {0.003: "colab_closed_loop_track_jitter0.003.csv", 0.01: "closed_loop_track_readout.csv",
             0.03: "colab_closed_loop_track_jitter0.03.csv"}
    drift = {0.001: "cl_henon_track_drift_0.001.log", 0.003: "cl_henon_track_drift_0.003.log",
             0.01: "colab_cl_henon_track_drift1.csv", 0.03: "colab_cl_henon_track_drift3.csv"}

    def slow2(name):
        if name.endswith(".csv"):
            return float(controllers(name)["ring mask_slow2, PIN detectors"]["tracking_nmse"])
        import re
        for line in open(os.path.join(RES, name)):
            if line.startswith("ring mask_slow2, PIN detectors"):
                return float(re.search(r"tracking_nmse (\S+)", line).group(1))
    esn = float(rows["ESN, 31 nodes"]["tracking_nmse"])
    for series, color, label in ((white, AQUA, "white jitter"), (drift, ORANGE, "slow drift (1000 symbols)")):
        xs = sorted(series)
        ax.plot([x * 100 for x in xs], [slow2(series[x]) for x in xs], color=color, marker="o", markersize=4,
                markeredgecolor="white", markeredgewidth=0.8, label=f"2 ns ring, {label}", zorder=3)
    ref_line(ax, esn, "ESN, same size", "--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.1, 0.3, 1, 3], ["0.1", "0.3", "1", "3"])
    ax.minorticks_off()
    ax.set_yticks([3e-4, 1e-3, 3e-3, 1e-2], ["3e-4", "1e-3", "3e-3", "1e-2"])
    ax.set_xlabel("detuning jitter (% of half-linewidth)")
    ax.set_ylabel("tracking NMSE")
    ax.set_title("sensitivity to the detuning jitter", loc="left")
    ax.legend(loc="upper left")
    panel_tag(ax, "c")
    save(fig, "figS4_closed_loop")


# ---- supplementary: mask ablation, fibre link -----------------------------------------------

def figS1():
    labels = {"comb_uniform": "uniform\ncomb", "comb_static": "random\nspectrum",
              "comb_amplitude": "amplitude\nonly", "comb_phase": "phase\nonly", "comb_mask": "full mask"}
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.4))
    for ax, name, title, tag in zip(axes, ("robust_ablation_mg_summary.csv", "robust_ablation_ce_summary.csv"),
                                    ("Mackey-Glass", "channel equalisation"), "ab"):
        rows = {r["group"]: r for r in read(name)}
        groups = [g for g in labels if g in rows]
        ax.bar(range(len(groups)), [float(rows[g]["fresh_test"]) for g in groups], width=0.6, color=AQUA, zorder=2)
        ref_line(ax, float(rows[groups[0]]["baseline_esn_fresh"]), "ESN, same size", "--")
        ax.legend(loc="upper right")
        ax.set_yscale("log")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([labels[g] for g in groups], fontsize=6)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left")
        ax.minorticks_off()
        lo, hi = ax.get_ylim()
        ticks = [t for t in (2e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3) if lo <= t <= hi]
        ax.set_yticks(ticks, [f"{t:.0e}".replace("e-0", "e-") for t in ticks])
        panel_tag(ax, tag)
    axes[0].set_ylabel("test NMSE, realistic noise")
    fig.tight_layout()
    save(fig, "figS1_mask_ablation")


def figS2():
    rows = {r["group"]: r for r in read("colab_fiber17_summary.csv")}
    rows.update({r["group"]: r for r in read("colab_link17_summary.csv")})
    items = [("power", "power\n(1 GBd)", BLUE), ("comb_uniform", "uniform comb\n(1 GBd)", AQUA),
             ("comb_mask", "phase mask\n(1 GBd)", AQUA), ("mask_slow2", "2 ns symbols\n(0.5 GBd)", AQUA),
             ("link_silica", "silica ring\n(28 GBd)", ORANGE), ("link_sin", "Si$_3$N$_4$ ring\n(28 GBd)", ORANGE)]
    fig, ax = plt.subplots(figsize=(SINGLE * 1.3, 2.4))
    vals = [float(rows[g]["fresh_ser"]) for g, _, _ in items]
    ax.bar(range(len(items)), vals, width=0.6, color=[c for _, _, c in items], zorder=2)
    any_row = rows["comb_mask"]
    ref_line(ax, float(any_row["baseline_esn_fresh_ser"]), "ESN, same size", "--")
    ref_line(ax, float(any_row["baseline_linear_ser"]), "linear equaliser", ":")
    ref_line(ax, 3.8e-3, "hard-decision FEC limit", "-.")
    ax.legend(loc="upper right")
    ax.set_yscale("log")
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([n for _, n, _ in items], fontsize=5.8)
    ax.grid(axis="x", visible=False)
    ax.set_yticks([1e-3, 2e-3, 5e-3, 1e-2, 2e-2], ["1e-3", "2e-3", "5e-3", "1e-2", "2e-2"])
    ax.set_ylim(1e-3, 2.5e-2)
    ax.set_ylabel("symbol error rate (lower is better)")
    ax.set_title("28 GBd PAM-4 over 17 km SSMF (IM/DD)", loc="left")
    save(fig, "figS2_fibre_link")


# ---- figure S3: the software baseline vs the input noise it is given ----------------------------

def figS3():
    rows = read("esn_input_noise_sweep.csv")
    levels = ["0", "1e-4", "3e-4", "1e-3", "3e-3"]
    rings = {"mackey_glass_h10": [(float(r["fresh_test"]), "ring, phase mask", "--")
                                  for r in read("robust_noisy_mg_summary.csv") if r["group"] == "comb_mask"]
                                 + [(float(r["fresh_test"]), "ring, stronger drop coupling", ":")
                                    for r in read("floor_mg_pin_summary.csv") if r["group"] == "mask_drop"],
             "channel_eq": [(float(r["fresh_test"]), "ring, phase mask", "--")
                            for r in read("colab_robust_noisy_ce_long_summary.csv") if r["group"] == "comb_mask"]}
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.5))
    for ax, task, title, tag in ((axes[0], "mackey_glass_h10", "Mackey-Glass", "a"),
                                 (axes[1], "channel_eq", "channel equalisation (28 dB)", "b")):
        x = np.arange(len(levels))
        for model, color in (("ESN 31", AQUA), ("ESN 100", BLUE), ("ESN 300", ORANGE)):
            val = {r["input_noise"]: float(r["test_nmse"]) for r in rows if r["task"] == task and r["model"] == model}
            ax.plot(x, [val.get(lv, np.nan) for lv in levels], color=color, marker="o", markersize=4,
                    markeredgecolor="white", markeredgewidth=0.8, label=model, zorder=3)
        for y, label, style in rings[task]:
            ref_line(ax, y, label, style)
        ax.axvspan(len(levels) - 1.3, len(levels) - 0.7, color=GRID, alpha=0.6, zorder=0, linewidth=0)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(["0", "1e-4", "3e-4", "1e-3", "3e-3\n(paper)"])
        ax.minorticks_off()
        ax.set_xlabel("software input noise (rms)")
        ax.set_ylabel("test NMSE")
        ax.set_title(rf"$\bf{{{tag}}}$  {title}", loc="left")     # tag in the title: the top axis takes its place
    axes[0].set_yticks([1e-8, 1e-7, 1e-6, 1e-5, 1e-4], ["1e-8", "1e-7", "1e-6", "1e-5", "1e-4"])
    axes[0].set_ylim(top=8e-4)
    axes[1].set_yticks([3e-3, 4e-3, 5e-3, 6e-3], ["3e-3", "4e-3", "5e-3", "6e-3"])

    # c: the closed loop (Henon map, tracking), software baselines against their input noise
    ax = axes[2]
    loop = read("closed_loop_input_noise.csv")
    x = np.arange(len(levels))
    for model, color, marker, label in (("ESN, 31 nodes", AQUA, "o", None), ("NG-RC", INK, "s", "NG-RC")):
        val = {r["input_noise"]: float(r["tracking_nmse"]) for r in loop
               if r["task"] == "track" and r["controller"] == model}
        ax.plot(x, [val.get(lv, np.nan) for lv in levels], color=color, marker=marker, markersize=4,
                markeredgecolor="white", markeredgewidth=0.8, label=label, zorder=3)
    for model, label, style in (("ring comb_mask, PIN detectors", "ring, phase mask", "--"),
                                ("ring mask_slow2, PIN detectors", "ring, 2 ns symbols", "-.")):
        for r in loop:
            if r["task"] == "track" and r["controller"] == model:
                ref_line(ax, float(r["tracking_nmse"]), label, style)
    ax.axvspan(len(levels) - 1.3, len(levels) - 0.7, color=GRID, alpha=0.6, zorder=0, linewidth=0)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["0", "1e-4", "3e-4", "1e-3", "3e-3\n(paper)"])
    ax.minorticks_off()
    ax.set_xlabel("software input noise (rms)")
    ax.set_ylabel("tracking NMSE")
    ax.set_yticks([1e-8, 1e-6, 1e-4, 1e-2], ["1e-8", "1e-6", "1e-4", "1e-2"])
    ax.set_title(r"$\bf{c}$  closed loop, tracking", loc="left")
    # top axis: the input noise read as the quantisation noise of an ADC whose full scale is the symbol range,
    # rms = 2^-N / sqrt(12), so N = log2(1 / (rms sqrt(12))) effective bits
    bits = ["∞"] + [f"{np.log2(1 / (float(lv) * np.sqrt(12))):.1f}" for lv in levels[1:]]
    for a in axes:
        top = a.secondary_xaxis("top")
        top.set_xticks(np.arange(len(levels)), bits)
        top.tick_params(labelsize=6, colors=INK2)
        top.set_xlabel("equivalent ADC bits", fontsize=6.5, color=INK2)
    # one legend for the three panels: the software models, then the rings (gray reference lines)
    seen = {}
    for a in axes:
        for h, lab in zip(*a.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    order = ["ESN 31", "ESN 100", "ESN 300", "NG-RC", "ring, phase mask", "ring, stronger drop coupling",
             "ring, 2 ns symbols"]
    fig.legend([seen[k] for k in order if k in seen], [k for k in order if k in seen], loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
    fig.tight_layout()
    save(fig, "figS3_esn_input_noise")


# ---- figure 9: ring arrays vs software reservoirs as the readout grows ---------------------------

def fig9():
    rows = read("scaling_mg.csv")
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    for model, color, style, label, names in (
            ("rings (shared RIN)", BLUE, "-", "rings, one laser feeds all", "rings"),
            ("ESN", ORANGE, "-", "ESN, input noise 3e-3 (paper's setting)", "nodes"),
            ("ESN, input noise 1e-3 (series 0)", ORANGE, "--", "ESN, input noise 1e-3 (one series)", None)):
        pts = sorted((int(r["size"]), float(r["test_nmse"])) for r in rows if r["model"] == model)
        x = [31 * s if names == "rings" else s for s, _ in pts]
        y = [v for _, v in pts]
        ax.plot(x, y, color=color, linestyle=style, marker="o", markersize=4, markeredgecolor="white",
                markeredgewidth=0.8, label=label, zorder=3)
        for (s, _), xi, yi in zip(pts, x, y) if names else ():
            ax.annotate(f"{s}", (xi, yi), textcoords="offset points", xytext=(0, 5 if names == "rings" else -10),
                        ha="center", fontsize=6, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_xticks([30, 100, 300, 1000], ["30", "100", "300", "1000"])
    ax.set_yticks([1e-6, 3e-6, 1e-5, 3e-5, 1e-4], ["1e-6", "3e-6", "1e-5", "3e-5", "1e-4"])
    ax.set_xlabel("comb lines read out (rings) or reservoir nodes (ESN)")
    ax.set_ylabel("test NMSE, Mackey-Glass")
    ax.set_title("readout size, 16k training symbols", loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig9_scaling")


FIGURES = {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6, "7": fig7, "9": fig9, "S1": figS1, "S2": figS2, "S3": figS3, "S4": figS4}

if __name__ == "__main__":
    for key in (sys.argv[1:] or FIGURES):
        FIGURES[key]()
