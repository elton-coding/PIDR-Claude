"""Draw the PIDR-Net refiner architecture (field response physics prior + residual refinement network).

Thumbnails come from pinn_refiner_thumbnails.npz (Sionna RT test map 1, gain channel in dB), produced from
checkpoints/sr/refiner_fr/FR/refiner.pth.  Run:  python paper/figures/draw_pinn_refiner.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT_DIR = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})

C = {
    "ink": "#25313C",
    "muted": "#5F6B76",
    "panel": "#FBEFD9",
    "panel_edge": "#C9A46A",
    "phys": "#E9DCC6",
    "phys_edge": "#A5783F",
    "cbr": "#BFE3E6",
    "cbr_edge": "#4F9AA3",
    "down": "#F6E7C8",
    "down_edge": "#C3A15A",
    "up": "#F5D3C4",
    "up_edge": "#C0806A",
    "interp": "#F8E8A8",
    "interp_edge": "#BFA34A",
    "out": "#EFEFEF",
    "out_edge": "#9A9A9A",
    "norm": "#E4F0E1",
    "norm_edge": "#7FA477",
    "add": "#FFFFFF",
    "skip": "#3478B9",
}


def box(ax, x, y, w, h, text, fill, edge, size=7.2, bold=False, rot=0, lw=1.0, rounding=0.08, color=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={rounding}",
                                facecolor=fill, edgecolor=edge, linewidth=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, rotation=rot,
            fontweight="bold" if bold else "normal", color=color or C["ink"], zorder=4, linespacing=1.25)


def arrow(ax, p, q, color=None, lw=1.0, cs="arc3", style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, linewidth=lw, color=color or C["ink"],
                                 connectionstyle=cs, shrinkA=0, shrinkB=0, zorder=2))


def line(ax, xs, ys, color=None, lw=1.0):
    ax.plot(xs, ys, color=color or C["ink"], lw=lw, zorder=2, solid_capstyle="butt")


def thumb(ax, img, x, y, s, cmap="jet", vmin=None, vmax=None, stack=0, stack_colors=None):
    """Image thumbnail with optional stacked frames behind it (multi-channel tensors)."""
    for k in range(stack, 0, -1):
        col = (stack_colors or ["#D9D9D9"] * stack)[k - 1]
        ax.add_patch(Rectangle((x + 0.09 * k, y + 0.09 * k), s, s, facecolor=col, edgecolor="#8C8C8C", lw=0.5, zorder=2))
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, extent=(x, x + s, y, y + s), origin="upper", zorder=3,
              interpolation="nearest", aspect="auto")
    ax.add_patch(Rectangle((x, y), s, s, fill=False, edgecolor="#8C8C8C", lw=0.5, zorder=4))


def main() -> None:
    d = np.load(OUT_DIR / "pinn_refiner_thumbnails.npz")
    vmin, vmax = np.nanmin(d["gt"]), np.nanmax(d["gt"])
    sparse = np.ma.masked_invalid(d["sparse"])
    coords = np.linspace(0, 1, 32)[None, :] * np.ones((32, 1))

    fig, ax = plt.subplots(figsize=(10.4, 6.3))
    ax.set_xlim(0, 20.8)
    ax.set_ylim(0, 12.6)
    ax.axis("off")

    # ---------------- inputs (left column) ----------------
    s = 1.55
    xin = 0.25
    inputs = [
        (9.75, d["diffusion"], "jet", 2, r"$\widehat{\mathbf{X}}^{\mathrm{diff}}$ coarse" + "\nreconstruction (3 ch)"),
        (6.75, sparse, "jet", 0, r"Measurements $\widetilde{\mathbf{X}}^{I}$," + "\n" + r"mask $\mathbf{M}^{I}$ (1 ch)"),
        (1.05, coords, "Greys", 1, "Coordinates\n" + r"$\mathbf{C}$ (2 ch)"),
    ]
    for y, img, cmap, st, label in inputs:
        if cmap == "jet":
            thumb(ax, img, xin, y, s, cmap, vmin, vmax, stack=st)
        else:
            thumb(ax, img, xin, y, s, cmap, 0, 1, stack=st, stack_colors=["#F3E3BF"])
        ax.text(xin + s / 2, y - 0.18, label, ha="center", va="top", fontsize=7.4, color=C["ink"], linespacing=1.15)

    # ---------------- physics prior panel (top) ----------------
    px, py, pw, ph = 3.0, 8.05, 17.55, 4.35
    ax.add_patch(FancyBboxPatch((px, py), pw, ph, boxstyle="round,pad=0.02,rounding_size=0.35",
                                facecolor=C["panel"], edgecolor=C["panel_edge"], lw=1.1, zorder=1))
    ax.text(px + 0.45, py + ph - 0.32, "Field Response Physics Prior (per map, no trainable parameters)",
            ha="left", va="center", fontsize=9.0, fontweight="bold", color=C["ink"])

    bw, bh, by = 3.15, 1.55, 9.55
    xs = [3.65, 7.45, 11.25]
    box(ax, xs[0], by, bw, bh, "Wavenumber spectrum\n" + r"$\Psi\propto|\mathcal{F}_{2N}\{\hat h^{\mathrm{diff}}\}|^{2}$",
        C["phys"], C["phys_edge"], size=7.3)
    box(ax, xs[1], by, bw, bh, "Plane-wave covariance\n" + r"$\Gamma(\mathbf{d})=\mathcal{F}^{-1}_{2N}\{\Psi\}(\mathbf{d})$",
        C["phys"], C["phys_edge"], size=7.3)
    box(ax, xs[2], by, bw + 0.5, bh, "LMMSE from measurements\n"
        + r"$\hat h^{\mathrm{fr}}=\mathbf{\Gamma}_{q,\mathcal{A}}(\mathbf{\Gamma}_{\mathcal{A}\mathcal{A}}+\delta^{2}\mathbf{I})^{-1}\mathbf{y}$",
        C["phys"], C["phys_edge"], size=7.3)
    arrow(ax, (xs[0] + bw, by + bh / 2), (xs[1], by + bh / 2))
    arrow(ax, (xs[1] + bw, by + bh / 2), (xs[2], by + bh / 2))
    ax.text(xs[0] + bw / 2, by - 0.32, r"$\hat h^{\mathrm{diff}}=\hat X^{\mathrm{re}}+j\hat X^{\mathrm{im}}$",
            ha="center", va="center", fontsize=7.0, color=C["muted"])
    ax.text(xs[2] + (bw + 0.5) / 2, by - 0.32, r"$\mathbf{y}$: measured complex CSI on $\mathcal{A}_{\mathrm{in}}$",
            ha="center", va="center", fontsize=7.0, color=C["muted"])

    # outputs of the physics prior
    ts = 1.3
    ox = 16.05
    thumb(ax, d["physics"], ox, 10.55, ts, "jet", vmin, vmax, stack=2)
    thumb(ax, d["logvar"], ox, 8.55, ts, "viridis", -1, 1)
    ax.text(ox + ts + 0.35, 10.55 + ts / 2, r"Physics estimate" + "\n" + r"$\widehat{\mathbf{X}}^{\mathrm{fr}}$ (3 ch)",
            ha="left", va="center", fontsize=7.4)
    ax.text(ox + ts + 0.35, 8.55 + ts / 2, "Confidence " + r"$\mathbf{V}$" + " (1 ch)" + "\nposterior variance", ha="left", va="center", fontsize=7.4)
    arrow(ax, (xs[2] + bw + 0.5, by + bh / 2 + 0.25), (ox - 0.05, 10.55 + ts / 2), cs="arc3,rad=-0.15")
    arrow(ax, (xs[2] + bw + 0.5, by + bh / 2 - 0.25), (ox - 0.05, 8.55 + ts / 2), cs="arc3,rad=0.15")

    # input routing into the physics panel
    line(ax, [xin + s + 0.18, 2.55], [9.75 + s / 2, 9.75 + s / 2])
    line(ax, [2.55, 2.55], [9.75 + s / 2, by + bh / 2])
    arrow(ax, (2.55, by + bh / 2), (xs[0], by + bh / 2))
    line(ax, [xin + s, 2.3], [6.75 + s / 2, 6.75 + s / 2])
    line(ax, [2.3, 2.3], [6.75 + s / 2, 8.75])
    line(ax, [2.3, xs[2] + (bw + 0.5) / 2], [8.75, 8.75])
    arrow(ax, (xs[2] + (bw + 0.5) / 2, 8.75), (xs[2] + (bw + 0.5) / 2, by - 0.55))

    # ---------------- concatenation bar ----------------
    cx, cw, cy, chh = 2.85, 0.55, 0.85, 6.6
    box(ax, cx, cy, cw, chh, "10-channel concatenation", C["cbr"], C["cbr_edge"], size=7.6, rot=90, rounding=0.12)
    # diffusion estimate and mask to the bar
    line(ax, [2.55, 2.55], [9.75 + s / 2, 6.1])
    arrow(ax, (2.55, 6.1), (cx, 6.1))
    arrow(ax, (2.3, 5.3), (cx, 5.3))
    line(ax, [2.3, 2.3], [6.75 + s / 2, 5.3])
    line(ax, [xin + s + 0.18, 2.3], [1.05 + s / 2, 1.05 + s / 2])
    line(ax, [2.3, 2.3], [1.05 + s / 2, 2.0])
    arrow(ax, (2.3, 2.0), (cx, 2.0))
    # physics outputs down to the bar (routed along the panel bottom)
    line(ax, [ox + ts / 2, ox + ts / 2], [8.55, 7.75])
    line(ax, [3.65, ox + ts / 2], [7.75, 7.75])
    line(ax, [3.65, 3.65], [7.75, 4.4])
    arrow(ax, (3.65, 4.4), (cx + cw, 4.4), style="-|>")
    ax.text(4.3, 7.78, r"$\widehat{\mathbf{X}}^{\mathrm{fr}}$, $\mathbf{V}$", fontsize=6.8, color=C["muted"], ha="left", va="bottom")

    # ---------------- refinement backbone ----------------
    yb, hb = 1.05, 5.9
    col_w = 0.62
    blocks = [
        # x, w, y, h, text, fill, edge, channels
        (4.35, col_w, 1.15, 5.7, "Conv-BN-ReLU", "cbr", 10),
        (5.25, col_w, 1.6, 4.8, "Conv-BN-ReLU", "cbr", 32),
        (6.55, col_w, 1.9, 4.2, r"3$\times$3 Conv, stride 2", "down", 32),
        (7.85, col_w, 2.3, 3.4, "Conv-BN-ReLU", "cbr", 64),
        (8.75, col_w, 2.3, 3.4, "Conv-BN-ReLU", "cbr", 64),
        (10.05, col_w, 2.3, 3.4, "Bilinear upsample", "interp", 64),
        (10.95, col_w, 1.9, 4.2, r"3$\times$3 Conv", "up", 32),
        (12.25, col_w, 1.6, 4.8, "Conv-BN-ReLU", "cbr", 64),
        (13.15, col_w, 1.6, 4.8, "Conv-BN-ReLU", "cbr", 32),
        (14.45, col_w, 1.9, 4.2, r"1$\times$1 Conv (zero init.)", "out", 32),
    ]
    for x, w, y, h, text, key, ch in blocks:
        box(ax, x, y, w, h, text, C[key], C[key + "_edge"], size=7.0, rot=90, rounding=0.12)
        ax.text(x + w / 2, y + h + 0.2, str(ch), ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    mid = 4.0
    arrow(ax, (cx + cw, mid), (4.35, mid))
    for (x0, w0, *_), (x1, *_r) in zip(blocks[:-1], blocks[1:]):
        if x1 - (x0 + w0) > 0.4:
            arrow(ax, (x0 + w0, mid), (x1, mid))
    # group frames
    for x0, x1, label in [(4.2, 6.02, "Encoder"), (7.7, 9.52, "Bottleneck"), (12.1, 13.92, "Decoder")]:
        ax.add_patch(Rectangle((x0, 0.95), x1 - x0, 6.15, fill=False, ls="--", lw=1.0, edgecolor=C["ink"], zorder=1))
        ax.text((x0 + x1) / 2, 0.72, label, ha="center", va="top", fontsize=7.8)
    ax.text(6.86, 1.62, "Down\nsample", ha="center", va="top", fontsize=7.2)
    ax.text(10.36 + 0.45, 1.62, "Up sample,\nproject", ha="center", va="top", fontsize=7.2)
    # skip connection
    line(ax, [5.56, 5.56], [1.6, 0.25], color=C["skip"])
    line(ax, [5.56, 11.85], [0.25, 0.25], color=C["skip"])
    arrow(ax, (11.85, 0.25), (11.85, mid - 0.05), color=C["skip"])
    ax.text(8.7, 0.1, "Skip connection", ha="center", va="top", fontsize=7.4, color=C["skip"])

    # residual addition on the physics estimate
    ax_c = (15.75, mid)
    arrow(ax, (14.45 + col_w, mid), (ax_c[0] - 0.22, mid))
    ax.add_patch(plt.Circle(ax_c, 0.22, facecolor="white", edgecolor=C["ink"], lw=1.0, zorder=4))
    ax.text(*ax_c, "+", ha="center", va="center", fontsize=10, zorder=5)
    ax.text(15.2, mid + 0.18, r"$\Delta$", ha="center", va="bottom", fontsize=7.6)
    line(ax, [ox + ts + 0.25, 17.6], [10.55 + ts / 2 - 0.25, 10.55 + ts / 2 - 0.25], color=C["muted"])
    line(ax, [17.6, 17.6], [10.55 + ts / 2 - 0.25, 7.45], color=C["muted"])
    line(ax, [15.75, 17.6], [7.45, 7.45], color=C["muted"])
    arrow(ax, (15.75, 7.45), (15.75, mid + 0.22), color=C["muted"])
    ax.text(15.85, 6.6, r"$\widehat{\mathbf{X}}^{\mathrm{fr}}$", ha="left", va="center", fontsize=7.4, color=C["muted"])
    box(ax, 16.45, 1.9, col_w, 4.2, r"clip $[-1,1]$, restore measured", C["out"], C["out_edge"], size=7.0, rot=90, rounding=0.12)
    arrow(ax, (ax_c[0] + 0.22, mid), (16.45, mid))
    box(ax, 17.45, 1.9, col_w, 4.2, "Inverse normalization", C["norm"], C["norm_edge"], size=7.0, rot=90, rounding=0.12)
    arrow(ax, (16.45 + col_w, mid), (17.45, mid))
    thumb(ax, d["refined"], 18.75, mid - 0.78, 1.55, "jet", vmin, vmax, stack=2)
    arrow(ax, (17.45 + col_w, mid), (18.75, mid))
    ax.text(18.75 + 0.78, mid - 1.0, "Refined CSI map", ha="center", va="top", fontsize=7.6)

    fig.savefig(OUT_DIR / "pinn_refiner_architecture.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT_DIR.parent.parent / "runs" / "pinn_refiner_architecture_preview.png", dpi=160, bbox_inches="tight")
    print("saved", OUT_DIR / "pinn_refiner_architecture.pdf")


if __name__ == "__main__":
    main()
