from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path(__file__).resolve().parent

COLORS = {
    "input": "#F7E6D2",
    "input_edge": "#C98645",
    "encoder": "#DCEAF6",
    "encoder_edge": "#4A7FA7",
    "bottleneck": "#E1F0E5",
    "bottleneck_edge": "#5D8B69",
    "decoder": "#DDF1EE",
    "decoder_edge": "#3E8C82",
    "attention": "#F1E5D4",
    "attention_edge": "#A5783F",
    "output": "#F4DEDE",
    "output_edge": "#A85B5B",
    "time": "#FFF0C7",
    "time_edge": "#B78A28",
    "ink": "#25313C",
    "muted": "#5F6B76",
    "skip": "#3478B9",
}


def block(ax, x, y, w, h, title, detail, fill, edge, lw=1.25, title_size=8.2, detail_size=7.0):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.09",
        facecolor=fill,
        edgecolor=edge,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.62,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=4,
    )
    ax.text(
        x + w / 2,
        y + h * 0.27,
        detail,
        ha="center",
        va="center",
        fontsize=detail_size,
        color=COLORS["muted"],
        linespacing=1.15,
        zorder=4,
    )
    return patch


def arrow(ax, start, end, color=None, lw=1.25, style="-|>", connectionstyle="arc3", zorder=2):
    a = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=9,
        linewidth=lw,
        color=color or COLORS["ink"],
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(a)
    return a


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(16.0, 6.15))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 20.0)
    ax.set_ylim(0, 8.1)
    ax.axis("off")

    ax.text(
        10.0,
        7.72,
        "Conditional Diffusion U-Net for Noise Prediction",
        ha="center",
        va="center",
        fontsize=14.2,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        10.0,
        7.34,
        "Residual encoder-decoder with bottleneck spatial self-attention",
        ha="center",
        va="center",
        fontsize=8.8,
        color=COLORS["muted"],
    )

    # Inputs converge to the seven-channel conditioning tensor.
    input_x, input_w, input_h = 0.12, 1.34, 0.92
    block(ax, input_x, 5.48, input_w, input_h, r"Noisy state $x_t$", r"$3\times32\times32$", COLORS["input"], COLORS["input_edge"], title_size=7.2, detail_size=6.5)
    block(ax, input_x, 4.24, input_w, input_h, r"Sparse CSI $\widetilde{Y}$", r"$3\times32\times32$", COLORS["input"], COLORS["input_edge"], title_size=7.2, detail_size=6.5)
    block(ax, input_x, 3.00, input_w, input_h, r"Mask $M$", r"$1\times32\times32$", COLORS["input"], COLORS["input_edge"], title_size=7.2, detail_size=6.5)
    block(ax, 1.70, 4.00, 1.08, 1.36, "Concat", r"$7\times32\times32$", "#FAF4EC", COLORS["input_edge"], title_size=7.3, detail_size=6.4)
    for yy in (5.94, 4.70, 3.46):
        arrow(ax, (1.46, yy), (1.70, 4.68), color=COLORS["input_edge"], lw=0.95)

    # Single left-to-right feature path. All shapes correspond to code/model.py.
    y, h, w = 4.00, 1.36, 1.02
    specs = [
        (3.02, "Input\nprojection", "$3\\times3$ Conv\n$7\\rightarrow64$", "encoder"),
        (4.28, "Encoder\nResBlock", "$64\\times32\\times32$", "encoder"),
        (5.54, "Downsample", "$3\\times3$, s=2\n$64\\rightarrow128$", "encoder"),
        (6.80, "Encoder\nResBlock", "$128\\times16\\times16$", "encoder"),
        (8.06, "Downsample", "$3\\times3$, s=2\n$128\\rightarrow256$", "encoder"),
        (9.32, "Mid\nResBlock", "$256\\times8\\times8$", "bottleneck"),
        (10.58, "Spatial\nMHA", "4 heads\n64 tokens", "attention"),
        (11.84, "Mid\nResBlock", "$256\\times8\\times8$", "bottleneck"),
        (13.10, "Upsample\n+ Conv", "bilinear + $3\\times3$\n$256\\rightarrow128$", "decoder"),
        (14.36, "Decoder\nResBlock", "concat: 256 ch\n$128\\times16\\times16$", "decoder"),
        (15.62, "Upsample\n+ Conv", "bilinear + $3\\times3$\n$128\\rightarrow64$", "decoder"),
        (16.88, "Decoder\nResBlock", "concat: 128 ch\n$64\\times32\\times32$", "decoder"),
        (18.14, "Output\nhead", "$3\\times3$ + $1\\times1$\n$64\\rightarrow3$", "output"),
    ]
    palette = {
        "encoder": (COLORS["encoder"], COLORS["encoder_edge"]),
        "bottleneck": (COLORS["bottleneck"], COLORS["bottleneck_edge"]),
        "attention": (COLORS["attention"], COLORS["attention_edge"]),
        "decoder": (COLORS["decoder"], COLORS["decoder_edge"]),
        "output": (COLORS["output"], COLORS["output_edge"]),
    }
    previous_right = 2.78
    for x, title, detail, kind in specs:
        fill, edge = palette[kind]
        block(ax, x, y, w, h, title, detail, fill, edge, title_size=6.8, detail_size=5.85)
        arrow(ax, (previous_right, y + h / 2), (x, y + h / 2), lw=1.05)
        previous_right = x + w

    block(ax, 19.28, 4.00, 0.68, 1.36, r"$\hat{\epsilon}_\theta$", r"$3\times32\times32$", COLORS["output"], COLORS["output_edge"], title_size=8.2, detail_size=4.7)
    arrow(ax, (19.16, y + h / 2), (19.28, y + h / 2), color=COLORS["output_edge"], lw=1.05)

    # Skip connections terminate at the decoder concatenations.
    arrow(ax, (4.79, 5.36), (17.39, 5.36), color=COLORS["skip"], lw=1.35, connectionstyle="arc3,rad=-0.18", zorder=1)
    ax.text(11.10, 6.98, "skip connection: 64 channels", ha="center", fontsize=7.2, color=COLORS["skip"])
    arrow(ax, (7.31, 5.36), (14.87, 5.36), color=COLORS["skip"], lw=1.35, connectionstyle="arc3,rad=-0.22", zorder=1)
    ax.text(11.10, 6.28, "skip connection: 128 channels", ha="center", fontsize=7.2, color=COLORS["skip"])

    # Time embedding and injection bus.
    block(ax, 0.82, 0.45, 2.25, 1.15, "Time embedding", "sin/cos + Linear--SiLU--Linear\n$t\\rightarrow256$", COLORS["time"], COLORS["time_edge"], title_size=7.6, detail_size=6.3)
    bus_y = 2.18
    ax.plot([1.95, 17.39], [bus_y, bus_y], color=COLORS["time_edge"], lw=1.05, linestyle=(0, (4, 3)), zorder=1)
    arrow(ax, (1.95, 1.60), (1.95, bus_y), color=COLORS["time_edge"], lw=1.0)
    for tx in (4.79, 7.31, 9.83, 12.35, 14.87, 17.39):
        arrow(ax, (tx, bus_y), (tx, y), color=COLORS["time_edge"], lw=0.9)
    ax.text(10.0, 1.88, "The 256-D time vector is added to every residual block", ha="center", fontsize=7.1, color="#8B681E")

    # Legend.
    arrow(ax, (13.20, 0.92), (14.00, 0.92), lw=1.15)
    ax.text(14.10, 0.92, "feature flow", va="center", fontsize=7.0, color=COLORS["muted"])
    arrow(ax, (15.55, 0.92), (16.35, 0.92), color=COLORS["skip"], lw=1.25)
    ax.text(16.45, 0.92, "skip connection", va="center", fontsize=7.0, color=COLORS["muted"])
    ax.plot([18.10, 18.90], [0.92, 0.92], color=COLORS["time_edge"], lw=1.05, linestyle=(0, (4, 3)))
    ax.text(19.00, 0.92, "time conditioning", va="center", fontsize=7.0, color=COLORS["muted"])

    plt.subplots_adjust(left=0.012, right=0.988, top=0.985, bottom=0.02)
    for suffix in ("svg", "pdf", "png"):
        path = OUT_DIR / f"conditional_diffusion_unet_architecture.{suffix}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
