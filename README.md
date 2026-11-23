# How far can a Kerr-microcomb reservoir computer go?

Simulation code and results for the paper *How far can a Kerr-microcomb reservoir computer go? Spectral-phase input masks,
the noise floor and a fair software reference* (Changxu Du, University of Shanghai for Science and Technology).

The paper asks how well a Kerr-microresonator reservoir computer can do under realistic noise, what limits it, and how it
compares fairly with software. In short:

- **Spectral-phase input mask.** Giving every line of an input comb a random complex bias and modulation weight puts the
  resonator in a coherently seeded regime with reproducible output. Under equal conditions it beats single-pump encodings and
  a uniformly modulated comb by one to two orders of magnitude on Mackey–Glass prediction. The randomness of the line phases is
  what matters.
- **Noise floor.** With ideal detectors about half of the error floor is detuning jitter and half is shot noise. The jitter
  cannot be removed in the readout, so the laser's intrinsic linewidth must be below about 5 kHz for this device.
- **Software reference.** The best single resonator matches a 31-node echo state network fed through an ADC with about 7
  effective bits; with the resonator's own 31 readout channels quantized as well, software stays ahead at equal bit depth.

## Layout

| Path | Contents |
|---|---|
| `combrc/` | the simulator: Lugiato–Lefever equation in modal form, RK4IP integrator (differentiable, PyTorch), noise models, readout, tasks |
| `experiments/` | one script per experiment; each script's docstring says what it does and how to run it |
| `experiments/figures.py` | all figures from the result files (`python experiments/figures.py`) |
| `experiments/make_table*.py` | Tables 1–3 (`--lang en` for the English paper) |
| `results/` | result summaries (CSV), logs and small trained models behind every table and figure |
| `figures/` | the figures as PDF and PNG |
| `paper_en/` | the English paper (LaTeX, pdflatex); `paper_en/make_arxiv.py` builds the arXiv package |
| `paper/` | the Chinese draft (LaTeX, XeLaTeX) |
| `tests/` | unit tests (`pytest`) |

The large spectral caches used by some scripts (`results/encodings_spectra/`, `results/operating_map_spectra/`, tens of GB)
are not included; the scripts regenerate them.

## Running

```
pip install -r requirements.txt
pytest tests
python experiments/compare_encodings.py --help
```

Most experiments were run on a GPU (`--device cuda`); the readout-quantization and shot-noise checks run on a CPU in minutes.

## Citation

If you use this code, please cite the paper (arXiv identifier to be added).

## Licence

MIT, see `LICENSE`.
