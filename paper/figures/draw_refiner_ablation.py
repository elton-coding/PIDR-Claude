"""Refiner ablation figure: NMSE reduction over the diffusion estimate on the three datasets.

Reads results/refiner_ablation_v2_<ds>/paper_protocol_n50/summary.csv (first 50 test maps,
5 diffusion sampling seeds; NP/FR averaged over 3 training seeds) and writes
paper/figures/refiner_ablation.pdf. Run from the repo root:
    python paper/figures/draw_refiner_ablation.py
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 8})

DATASETS = [("sr", "Sionna RT"), ("fd", "Field response"), ("el", "EM simulation")]
# (csv method, legend label, color)
VARIANTS = [
    ("Physics only", "Physics prior only", "#2a78d6"),
    ("NP", "Refiner without physics prior", "#eb6834"),
    ("FR", "PIDR-Net refiner", "#1baf7a"),
]
PROTOCOL = "protocol (first 50 test maps, 5 sampling seeds)"
INK, MUTED, GRID = "#1f1f1e", "#5f5e5a", "#d9d8d3"


def load(ds):
    path = os.path.join("results", f"refiner_ablation_v2_{ds}", "paper_protocol_n50", "summary.csv")
    rows = {r["method"]: r for r in csv.DictReader(open(path, encoding="utf-8")) if r["evaluation"] == PROTOCOL}
    base = float(rows["Diffusion only"]["NMSE_dB"])
    out = {}
    for method, _, _ in VARIANTS:
        r = rows[method]
        std = r["NMSE_dB_std_train_seeds"]
        out[method] = (base - float(r["NMSE_dB"]), float(std) if std else 0.0)
    return out


data = {ds: load(ds) for ds, _ in DATASETS}

fig, ax = plt.subplots(figsize=(3.5, 2.45))
x = np.arange(len(DATASETS))
width = 0.26
gap = 0.012  # small surface gap between adjacent bars
for i, (method, label, color) in enumerate(VARIANTS):
    vals = np.array([data[ds][method][0] for ds, _ in DATASETS])
    errs = np.array([data[ds][method][1] for ds, _ in DATASETS])
    pos = x + (i - 1) * width
    ax.bar(pos, vals, width - 2 * gap, color=color, label=label, zorder=3)
    if errs.any():
        ax.errorbar(pos, vals, yerr=errs, fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.8, zorder=4)
    for p, v, e in zip(pos, vals, errs):
        ax.text(p, v + e + 0.35, f"{v:.1f}", ha="center", va="bottom", fontsize=6.3, color=INK)

ax.set_xticks(x)
ax.set_xticklabels([name for _, name in DATASETS])
ax.set_ylabel("NMSE reduction over diffusion (dB)")
ax.set_ylim(0, 24)
ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(MUTED)
ax.tick_params(colors=INK, length=2.5)
ax.legend(loc="upper left", frameon=False, fontsize=6.8, handlelength=1.2, handleheight=0.9)
fig.tight_layout(pad=0.3)
fig.savefig("paper/figures/refiner_ablation.pdf")
fig.savefig("runs/refiner_ablation_preview.png", dpi=220)
for ds, _ in DATASETS:
    print(ds, {k: tuple(round(v, 2) for v in val) for k, val in data[ds].items()})
