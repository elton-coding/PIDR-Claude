"""Draw the single-MA channel mapping system used in the paper."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


HERE = Path(__file__).resolve().parent
INK = "#253341"
BLUE = "#406783"
BLUE_LIGHT = "#e8f0f5"
TEAL = "#177e7c"
TEAL_LIGHT = "#e8f4f1"
ORANGE = "#c47a26"
GRID = "#b8c8cf"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

fig, ax = plt.subplots(figsize=(3.45, 2.12))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_xlim(0, 3.45)
ax.set_ylim(0, 2.12)
ax.set_aspect("equal")
ax.axis("off")

# Base station with one transmit chain.
ax.add_patch(Rectangle((0.08, 0.83), 0.61, 0.45, facecolor=BLUE_LIGHT, edgecolor=BLUE, linewidth=1.0))
ax.text(0.385, 1.055, "BS", ha="center", va="center", color=INK, weight="bold", fontsize=8)
ax.text(0.385, 0.885, "Pilot source", ha="center", va="center", color=INK, fontsize=6.7)

# Candidate MA positions occupy a two-dimensional transmitter region.
gx, gy, side = 1.19, 0.55, 1.03
ax.add_patch(Rectangle((gx, gy), side, side, facecolor="#f8fbfc", edgecolor=BLUE, linewidth=1.05))
for k in range(1, 5):
    q = k * side / 5
    ax.plot((gx + q, gx + q), (gy, gy + side), color=GRID, linewidth=0.55)
    ax.plot((gx, gx + side), (gy + q, gy + q), color=GRID, linewidth=0.55)
for i in range(5):
    for j in range(5):
        ax.add_patch(Circle((gx + (i + 0.5) * side / 5, gy + (j + 0.5) * side / 5), 0.016,
                            facecolor=GRID, edgecolor="none"))

start = (gx + 1.5 * side / 5, gy + 1.5 * side / 5)
selected = (gx + 3.5 * side / 5, gy + 3.5 * side / 5)
ax.add_patch(FancyArrowPatch(start, selected, arrowstyle="-|>", mutation_scale=8,
                             linewidth=1.15, linestyle=(0, (3, 2)), color=TEAL,
                             connectionstyle="arc3,rad=0.2"))
ax.add_patch(Circle(selected, 0.057, facecolor=TEAL, edgecolor="white", linewidth=0.75, zorder=4))
ax.text(1.705, 1.83, "2D transmit MA region", ha="center", va="center", color=INK,
        weight="bold", fontsize=7.3)
ax.text(1.705, 0.36, r"$N\times N$ candidate positions", ha="center", va="center", color=INK,
        fontsize=7.0)
ax.text(selected[0] + 0.13, selected[1] + 0.15, r"$\mathbf{t}_{u,v}$", ha="left", va="center",
        color=TEAL, fontsize=8)

# A solid connection indicates the controlled transmit antenna; the radio link
# leaves the selected grid point and terminates at one conventional RX antenna.
ax.add_patch(FancyArrowPatch((0.71, 1.055), (1.16, 1.055), arrowstyle="-|>",
                             mutation_scale=9, linewidth=1.25, color=BLUE))
ax.add_patch(FancyArrowPatch((selected[0] + 0.065, selected[1]), (2.82, 1.12),
                             arrowstyle="-|>", mutation_scale=10, linewidth=1.35, color=ORANGE))
ax.text(2.56, 1.35, "Pilot signal", ha="center", va="center", color=ORANGE, fontsize=6.7)

# Single-antenna receiver. Its location changes only between offline samples.
rx_x, rx_y = 3.03, 1.12
ax.plot((rx_x, rx_x), (rx_y - 0.23, rx_y + 0.15), color=INK, linewidth=1.2)
ax.plot((rx_x - 0.16, rx_x, rx_x + 0.16), (rx_y + 0.02, rx_y + 0.15, rx_y + 0.02),
        color=INK, linewidth=1.2)
ax.add_patch(Circle((rx_x, rx_y - 0.25), 0.035, facecolor=INK, edgecolor="none"))
ax.text(rx_x, 0.72, "User RX", ha="center", va="center", color=INK, weight="bold", fontsize=7.3)
ax.text(rx_x, 0.56, "single antenna", ha="center", va="center", color=INK, fontsize=6.7)
ax.text(rx_x, 0.35, r"location $\mathbf{r}_s$", ha="center", va="center", color=INK, fontsize=7.0)

fig.savefig(HERE / "ma_system_model.pdf", transparent=True)
fig.savefig(HERE / "ma_system_model.png", dpi=300, transparent=False)
plt.close(fig)
