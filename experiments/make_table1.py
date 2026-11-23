"""Table 1 of the paper: test NMSE of every encoding and the baselines, from the result files.

Writes paper/tables/table1.tex (Chinese labels). Numbers are the paper's protocol: 256 random operating points
per encoding, choice on new runs, test NMSE as the mean over 6 further pairs of new runs, geometric mean over
3 series.

    python experiments/make_table1.py
"""
import csv
import sys
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
EN = "--lang" in sys.argv and sys.argv[sys.argv.index("--lang") + 1] == "en"
OUT = os.path.join(ROOT, "paper_en" if EN else "paper", "tables", "table1.tex")

COLUMNS = [  # (header, task, file)
    ("无噪声", "mackey_glass_h10", "robust_clean_summary.csv"),
    ("现实噪声", "mackey_glass_h10", "robust_noisy_mg_summary.csv"),
    ("无噪声", "channel_eq", "robust_clean_summary.csv"),
    ("现实噪声", "channel_eq", "colab_robust_noisy_ce_long_summary.csv"),
    ("无噪声", "narma10", "robust_clean_narma_summary.csv"),
    ("现实噪声", "narma10", "robust_noisy_narma_summary.csv"),
]
ROWS = [  # (label, group)
    (r"泵浦功率~\cite{shishavan2025}", "power"),
    (r"泵浦失谐~\cite{cuevas2025}", "detuning"),
    (r"泵浦相位~\cite{arabieh2026soliton}", "phase"),
    (r"泵浦功率，孤子起点", "power_soliton"),
    (r"泵浦失谐，孤子起点", "detuning_soliton"),
    (r"泵浦相位，孤子起点", "phase_soliton"),
    (r"整体调制光梳", "comb_uniform"),
    (r"谱相位掩模（本文）", "comb_mask"),
]


COLUMNS_EN = ["no noise", "realistic", "no noise", "realistic", "no noise", "realistic"]
ROWS_EN = [r"pump power~\cite{shishavan2025}", r"pump detuning~\cite{cuevas2025}", r"pump phase~\cite{arabieh2026soliton}",
           r"pump power, soliton start", r"pump detuning, soliton start", r"pump phase, soliton start",
           r"uniformly modulated comb", r"spectral-phase mask (this work)"]
CAPTION_EN = (r"\caption{Test error (NMSE, lower is better) of each encoding and the baselines. Each encoding draws 256 random "
              r"operating points over the same ranges; the point is chosen on the mean validation error of new runs and the mean "
              r"test error over six further pairs of new runs is reported, as the geometric mean over three series; the readout is "
              r"trained on one run and tested on another with independent noise. The best resonator encoding in each column is in "
              r"bold. The ESN has the same readout dimension as the resonator (31 nodes, the same delay taps) and uses a single "
              r"random reservoir; under realistic noise its input noise is $3\times10^{-3}$; stronger software baselines are in "
              r"Section~\ref{sec:software}. Channel equalisation under realistic noise uses test parts of 5000 symbols.}")


def fmt(x, bold=False):
    exp = math.floor(math.log10(x))
    man = x / 10 ** exp
    if round(man, 1) >= 10:
        man, exp = man / 10, exp + 1
    s = rf"{man:.1f}\times10^{{{exp}}}" if exp not in (-1, 0) else f"{x:.2f}" if x < 1 else f"{x:.1f}"
    s = s if exp not in (-1, 0) else s
    body = rf"\mathbf{{{s}}}" if bold else s
    return rf"${body}$"


def main():
    data, esn, lin = {}, [], []
    for _, task, name in COLUMNS:
        rows = {r["group"]: r for r in csv.DictReader(open(os.path.join(RES, name))) if r["task"] == task}
        data[(task, name)] = {g: float(r["fresh_test"]) for g, r in rows.items()}
        any_row = next(iter(rows.values()))
        esn.append(float(any_row["baseline_esn_fresh"]))
        lin.append(float(any_row["baseline_linear"]))
    best = {k: min(v[g] for _, g in ROWS) for k, v in data.items()}
    lines = [r"\begin{table*}[htbp]", r"\centering",
             r"\caption{各编码与基线的测试误差（NMSE，越小越好）。每种编码在相同范围内各抽 256 个随机工作点，"
             r"按新运行的平均验证误差选点，报告另外 6 组新运行的平均测试误差，再取 3 条序列的几何平均；"
             r"读出层训练在一次运行上、测试在另一次噪声独立的运行上。每列最好的微环编码用粗体标出。"
             r"ESN 与微环的读出维度相同（31 个节点、相同的延迟抽头），只用一个随机储备池，"
             r"现实噪声下给它的输入噪声为 $3\times10^{-3}$；更强的软件基线见第~\ref{sec:software}~节。"
             r"信道均衡在现实噪声下的测试段为 5000 个符号。}",
             r"\label{tab:encodings}", r"\small", r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{Mackey--Glass（10 步预测）} & \multicolumn{2}{c}{非线性信道均衡} "
             r"& \multicolumn{2}{c}{NARMA-10} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             "编码 & " + " & ".join(h for h, _, _ in COLUMNS) + r" \\", r"\midrule"]
    if EN:
        lines[2] = CAPTION_EN
        lines[4] = r"\footnotesize\setlength{\tabcolsep}{4pt}"
        lines[7] = (r" & \multicolumn{2}{c}{Mackey--Glass (10 steps)} & \multicolumn{2}{c}{channel equalisation} "
                    r"& \multicolumn{2}{c}{NARMA-10} \\")
        lines[9] = "encoding & " + " & ".join(COLUMNS_EN) + r" \\"
    for i, (label, group) in enumerate(ROWS):
        cells = [fmt(data[(t, n)][group], data[(t, n)][group] == best[(t, n)]) for _, t, n in COLUMNS]
        lines.append((ROWS_EN[i] if EN else label) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule",
              ("ESN (31 nodes)" if EN else r"ESN（31 节点）") + " & " + " & ".join(fmt(v) for v in esn) + r" \\",
              ("linear autoregression" if EN else r"线性自回归") + " & " + " & ".join(fmt(v) for v in lin) + r" \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("% generated by experiments/make_table1.py; do not edit by hand\n" + "\n".join(lines) + "\n")
    print("wrote", OUT)
    for (t, n), v in data.items():
        print(t, n, "mask", f"{v['comb_mask']:.2e}", "best other",
              min((x, g) for g, x in v.items() if g != "comb_mask"))


if __name__ == "__main__":
    main()
