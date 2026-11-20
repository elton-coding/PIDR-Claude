"""Draw the PIDR-Net refiner architecture (field response physics prior + residual refinement network).

Writes the paper figure (pinn_refiner_architecture.pdf) and an editable PowerPoint version
(pinn_refiner_architecture.pptx) from the same geometry. Every shape and text is recorded once and
rendered by both back ends; in the .pptx all boxes, arrows and texts are native, editable objects
(formulas are rich text with sub/superscripts), and only the CSI-map thumbnails are pictures.

Equation numbers are read from paper/pidr_net.aux, so compile the paper first and rerun this script
whenever the equations are renumbered.

Thumbnails come from pinn_refiner_thumbnails.npz (Sionna RT test map 1, gain channel in dB), produced from
checkpoints/sr/refiner_fr/FR/refiner.pth.  Run (repo root, needs python-pptx):
    python paper/figures/draw_pinn_refiner.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT_DIR = Path(__file__).resolve().parent
PAPER_DIR = OUT_DIR.parent
REPO = PAPER_DIR.parent
BUILD_DIR = REPO / "runs" / "pinn_refiner_pptx"  # thumbnail PNGs for the .pptx
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})

XMAX, YMAX = 21.0, 12.6  # drawing units

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


def equation_numbers() -> dict:
    """Map equation labels to their numbers in the compiled paper (from pidr_net.aux)."""
    aux = PAPER_DIR / "pidr_net.aux"
    if not aux.exists():
        raise SystemExit("paper/pidr_net.aux not found: compile the paper first (latexmk in paper/).")
    nums = {}
    for m in re.finditer(r"\\newlabel\{(eq:[^}]+)\}\{\{([^}]+)\}", aux.read_text(encoding="utf-8")):
        nums[m.group(1)] = m.group(2)
    return nums


class Scene:
    """Draws on a matplotlib axis and records every primitive for the PowerPoint export.

    ``ppt`` arguments are rich-text markup for PowerPoint: ``_{..}`` subscript, ``^{..}`` superscript,
    ``\\b{..}`` bold, ``\\i{..}`` italic, ``\\m{..}`` math font, ``\\{`` / ``\\}`` literal braces,
    newline = new paragraph. Without ``ppt`` the plain text is used.
    """

    def __init__(self, ax):
        self.ax = ax
        self.items = []

    def box(self, x, y, w, h, text, fill, edge, size=7.2, bold=False, rot=0, lw=1.0, rounding=0.08,
            color=None, ppt=None, z=3):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={rounding}",
                                         facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z))
        if text:
            self.ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, rotation=rot,
                         fontweight="bold" if bold else "normal", color=color or C["ink"], zorder=z + 1,
                         linespacing=1.25)
        self.items.append(dict(kind="box", x=x, y=y, w=w, h=h, ppt=ppt if ppt is not None else text, fill=fill,
                               edge=edge, size=size, bold=bold, rot=rot, lw=lw, rounding=rounding,
                               color=color or C["ink"]))

    def text(self, x, y, text, size=7.4, ha="center", va="center", color=None, bold=False, ppt=None):
        self.ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color or C["ink"],
                     fontweight="bold" if bold else "normal", linespacing=1.15, zorder=5)
        self.items.append(dict(kind="text", x=x, y=y, ppt=ppt if ppt is not None else text, size=size, ha=ha,
                               va=va, color=color or C["ink"], bold=bold))

    def arrow(self, p, q, color=None, lw=1.0, cs="arc3"):
        self.ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, linewidth=lw,
                                          color=color or C["ink"], connectionstyle=cs, shrinkA=0, shrinkB=0,
                                          zorder=2))
        self.items.append(dict(kind="arrow", p=p, q=q, color=color or C["ink"], lw=lw))

    def line(self, xs, ys, color=None, lw=1.0):
        self.ax.plot(xs, ys, color=color or C["ink"], lw=lw, zorder=2, solid_capstyle="butt")
        self.items.append(dict(kind="line", xs=list(xs), ys=list(ys), color=color or C["ink"], lw=lw))

    def rect(self, x, y, w, h, edge, lw=1.0, dashed=False, fill=None):
        self.ax.add_patch(Rectangle((x, y), w, h, fill=fill is not None, facecolor=fill, ls="--" if dashed else "-",
                                    lw=lw, edgecolor=edge, zorder=1 if dashed else 2))
        self.items.append(dict(kind="rect", x=x, y=y, w=w, h=h, edge=edge, lw=lw, dashed=dashed, fill=fill))

    def circle(self, center, r, text, size=10):
        self.ax.add_patch(plt.Circle(center, r, facecolor="white", edgecolor=C["ink"], lw=1.0, zorder=4))
        self.ax.text(*center, text, ha="center", va="center", fontsize=size, zorder=5)
        self.items.append(dict(kind="circle", center=center, r=r, text=text, size=size))

    def thumb(self, img, x, y, s, cmap="jet", vmin=None, vmax=None, stack=0, stack_colors=None):
        """Image thumbnail with optional stacked frames behind it (multi-channel tensors)."""
        for k in range(stack, 0, -1):
            col = (stack_colors or ["#D9D9D9"] * stack)[k - 1]
            self.rect(x + 0.09 * k, y + 0.09 * k, s, s, "#8C8C8C", lw=0.5, fill=col)
        self.ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, extent=(x, x + s, y, y + s), origin="upper", zorder=3,
                       interpolation="nearest", aspect="auto")
        self.ax.add_patch(Rectangle((x, y), s, s, fill=False, edgecolor="#8C8C8C", lw=0.5, zorder=4))
        self.items.append(dict(kind="image", x=x, y=y, s=s, img=img, cmap=cmap, vmin=vmin, vmax=vmax))


# ---------------------------------------------------------------- PowerPoint export
def parse_markup(s):
    """Split PowerPoint markup into runs of (text, style dict); '\n' becomes a paragraph break run."""
    runs = []

    def walk(t, style):
        i, buf = 0, ""

        def flush():
            nonlocal buf
            if buf:
                runs.append((buf, dict(style)))
                buf = ""

        while i < len(t):
            ch = t[i]
            if ch == "\\" and i + 1 < len(t) and t[i + 1] in "{}":
                buf += t[i + 1]
                i += 2
                continue
            tag = None
            if t.startswith(("\\b{", "\\i{", "\\m{"), i):
                tag, start = t[i + 1], i + 3
            elif t.startswith(("_{", "^{"), i):
                tag, start = t[i], i + 2
            if tag:
                depth, j = 1, start
                while depth:
                    if t[j] == "\\" and t[j + 1] in "{}":
                        j += 2
                        continue
                    depth += {"{": 1, "}": -1}.get(t[j], 0)
                    j += 1
                flush()
                sub = dict(style)
                sub.update({"b": {"bold": True}, "i": {"italic": True}, "m": {"math": True},
                            "_": {"base": -25000}, "^": {"base": 30000}}[tag])
                walk(t[start:j - 1], sub)
                i = j
                continue
            if ch == "\n":
                flush()
                runs.append(("\n", {}))
                i += 1
                continue
            buf += ch
            i += 1
        flush()

    walk(s, {})
    return runs


def mathtext_to_plain(s):
    """Fallback for labels given only as matplotlib text: strip $ and simple TeX."""
    s = s.replace("$", "")
    for a, b in [(r"\times", "×"), (r"\Delta", "Δ"), ("\\", "")]:
        s = s.replace(a, b)
    return s


def to_pptx(scene, path):
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.oxml.ns import qn
    from pptx.util import Emu, Inches, Pt

    unit = 0.625  # inches per drawing unit
    fscale = unit / 0.5  # the matplotlib figure uses 0.5 inch per unit
    prs = Presentation()
    EXL, EXR, EYB, EYT = -0.35, XMAX + 0.15, -0.6, YMAX + 0.15  # slide extent with margins (drawing units)
    prs.slide_width, prs.slide_height = Inches((EXR - EXL) * unit), Inches((EYT - EYB) * unit)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    shapes = slide.shapes
    X = lambda v: Inches((v - EXL) * unit)  # noqa: E731  (x position)
    Y = lambda v: Inches((EYT - v) * unit)  # noqa: E731  (y position, flipped)
    L = lambda v: Inches(v * unit)  # noqa: E731  (length)
    rgb = lambda h: RGBColor.from_string(h.lstrip("#").upper())  # noqa: E731
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    def fill_text(tf, markup, size, color, bold=False, align=PP_ALIGN.CENTER):
        tf.word_wrap = False
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        runs = parse_markup(markup if "$" not in markup else mathtext_to_plain(markup))
        para = tf.paragraphs[0]
        para.alignment = align
        for text, st in runs:
            if text == "\n":
                para = tf.add_paragraph()
                para.alignment = align
                continue
            r = para.add_run()
            r.text = text
            f = r.font
            f.size = Pt(round(size * fscale * 2) / 2)
            f.bold = bold or st.get("bold", False)
            f.italic = st.get("italic", False)
            f.name = "Cambria Math" if st.get("math") else "Arial"
            f.color.rgb = rgb(color)
            if "base" in st:
                f._element.set("baseline", str(st["base"]))

    def connector(x1, y1, x2, y2, color, lw, head):
        c = shapes.add_connector(MSO_CONNECTOR.STRAIGHT, X(x1), Y(y1), X(x2), Y(y2))
        c.line.color.rgb = rgb(color)
        c.line.width = Pt(lw)
        if head:
            ln = c.line._get_or_add_ln()
            tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
            ln.append(tail)
        return c

    for n, it in enumerate(scene.items):
        k = it["kind"]
        if k == "box":
            cx, cy, w, h = it["x"] + it["w"] / 2, it["y"] + it["h"] / 2, it["w"], it["h"]
            if it["rot"]:  # vertical box: draw it horizontal, then rotate (text reads bottom to top)
                w, h = h, w
            sh = shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, X(cx - w / 2), Y(cy + h / 2), L(w), L(h))
            sh.adjustments[0] = min(0.5, it["rounding"] / min(w, h))
            if it["rot"]:
                sh.rotation = 270
            sh.fill.solid()
            sh.fill.fore_color.rgb = rgb(it["fill"])
            sh.line.color.rgb = rgb(it["edge"])
            sh.line.width = Pt(it["lw"])
            sh.shadow.inherit = False
            tf = sh.text_frame
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            if it["ppt"]:
                fill_text(tf, it["ppt"], it["size"], it["color"], it["bold"])
        elif k == "text":
            lines = it["ppt"].count("\n") + 1
            h = lines * it["size"] * 1.3 / 72 / 0.5  # in drawing units (figure scale)
            w = 8.0
            xl = {"left": it["x"], "center": it["x"] - w / 2, "right": it["x"] - w}[it["ha"]]
            ytop = {"top": it["y"], "center": it["y"] + h / 2, "bottom": it["y"] + h}[it["va"]]
            tb = shapes.add_textbox(X(xl), Y(ytop), L(w), L(h))
            tb.text_frame.vertical_anchor = {"top": MSO_ANCHOR.TOP, "center": MSO_ANCHOR.MIDDLE,
                                             "bottom": MSO_ANCHOR.BOTTOM}[it["va"]]
            align = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[it["ha"]]
            fill_text(tb.text_frame, it["ppt"], it["size"], it["color"], it["bold"], align)
        elif k == "arrow":
            connector(*it["p"], *it["q"], it["color"], it["lw"], head=True)
        elif k == "line":
            for (x1, y1), (x2, y2) in zip(zip(it["xs"], it["ys"]), zip(it["xs"][1:], it["ys"][1:])):
                connector(x1, y1, x2, y2, it["color"], it["lw"], head=False)
        elif k == "rect":
            sh = shapes.add_shape(MSO_SHAPE.RECTANGLE, X(it["x"]), Y(it["y"] + it["h"]), L(it["w"]), L(it["h"]))
            sh.shadow.inherit = False
            if it["fill"]:
                sh.fill.solid()
                sh.fill.fore_color.rgb = rgb(it["fill"])
            else:
                sh.fill.background()
            sh.line.color.rgb = rgb(it["edge"])
            sh.line.width = Pt(it["lw"])
            if it["dashed"]:
                from pptx.enum.dml import MSO_LINE_DASH_STYLE
                sh.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        elif k == "circle":
            (cx, cy), r = it["center"], it["r"]
            sh = shapes.add_shape(MSO_SHAPE.OVAL, X(cx - r), Y(cy + r), L(2 * r), L(2 * r))
            sh.shadow.inherit = False
            sh.fill.solid()
            sh.fill.fore_color.rgb = rgb("#FFFFFF")
            sh.line.color.rgb = rgb(C["ink"])
            sh.line.width = Pt(1.0)
            sh.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
            fill_text(sh.text_frame, it["text"], it["size"], C["ink"])
        elif k == "image":
            img = np.ma.masked_invalid(it["img"])
            big = np.kron(img.filled(np.nan), np.ones((10, 10)))  # nearest-neighbour upscale: no smoothing in PPT
            png = BUILD_DIR / f"thumb_{n:02d}.png"
            plt.imsave(png, np.ma.masked_invalid(big), cmap=it["cmap"], vmin=it["vmin"], vmax=it["vmax"])
            shapes.add_picture(str(png), X(it["x"]), Y(it["y"] + it["s"]), L(it["s"]), L(it["s"]))
            fr = shapes.add_shape(MSO_SHAPE.RECTANGLE, X(it["x"]), Y(it["y"] + it["s"]), L(it["s"]), L(it["s"]))
            fr.fill.background()
            fr.shadow.inherit = False
            fr.line.color.rgb = rgb("#8C8C8C")
            fr.line.width = Pt(0.5)
    # drop the theme style references python-pptx adds (shadow/effect, white font); colors are explicit
    for el in list(slide.shapes._spTree.iter(qn("p:style"))):
        el.getparent().remove(el)
    prs.save(path)


# ---------------------------------------------------------------- figure
def main() -> None:
    eq = equation_numbers()
    E = lambda label: f"({eq[label]})"  # noqa: E731
    d = np.load(OUT_DIR / "pinn_refiner_thumbnails.npz")
    vmin, vmax = np.nanmin(d["gt"]), np.nanmax(d["gt"])
    sparse = np.ma.masked_invalid(d["sparse"])
    coords = np.linspace(0, 1, 32)[None, :] * np.ones((32, 1))

    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    ax.axis("off")
    S = Scene(ax)

    # ---------------- inputs (left column) ----------------
    s = 1.55
    xin = 0.25
    inputs = [
        (9.75, d["diffusion"], "jet", 2, r"$\widehat{\mathbf{X}}^{\mathrm{diff}}$ coarse" + "\nreconstruction (3 ch)",
         "\\m{\\b{X\u0302}^{diff}} coarse\nreconstruction (3 ch)"),
        (6.75, sparse, "jet", 0, r"Measurements $\widetilde{\mathbf{X}}^{I}$," + "\n" + r"mask $\mathbf{M}^{I}$ (1 ch)",
         "Measurements \\m{\\b{X\u0303}^{\\i{I}}},\nmask \\m{\\b{M}^{\\i{I}}} (1 ch)"),
        (1.05, coords, "Greys", 1, "Coordinates\n" + r"$\mathbf{C}$ (2 ch), " + E("eq:coordinate_tensor"),
         "Coordinates\n\\m{\\b{C}} (2 ch), " + E("eq:coordinate_tensor")),
    ]
    for y, img, cmap, st, label, ppt in inputs:
        if cmap == "jet":
            S.thumb(img, xin, y, s, cmap, vmin, vmax, stack=st)
        else:
            S.thumb(img, xin, y, s, cmap, 0, 1, stack=st, stack_colors=["#F3E3BF"])
        S.text(xin + s / 2, y - 0.18, label, va="top", ppt=ppt)

    # ---------------- physics prior panel (top) ----------------
    px, py, pw, ph = 3.0, 8.05, 17.75, 4.35
    S.box(px, py, pw, ph, "", C["panel"], C["panel_edge"], lw=1.1, rounding=0.35, z=1)
    S.text(px + 0.45, py + ph - 0.32, "Field Response Physics Prior (per map, no trainable parameters)",
           size=9.0, ha="left", bold=True)

    bh, by = 1.55, 9.55
    xs, ws = [3.3, 7.2, 11.1], [3.5, 3.5, 4.05]
    steps = [
        ("Wavenumber spectrum " + E("eq:fr_spectrum"),
         r"$\Psi\propto|\mathcal{F}_{2N}\{\hat h^{\mathrm{diff}}\}|^{2}$",
         "\\m{Ψ ∝ |ℱ_{2\\i{N}}\\{\\i{h\u0302}^{diff}\\}|^{2}}"),
        ("Plane-wave covariance " + E("eq:fr_kernel"),
         r"$\Gamma(\mathbf{d})=\mathcal{F}^{-1}_{2N}\{\Psi\}(\mathbf{d})$",
         "\\m{Γ(\\b{d}) = ℱ^{−1}_{2\\i{N}}\\{Ψ\\}(\\b{d})}"),
        ("LMMSE from measurements " + E("eq:fr_lmmse"),
         r"$\hat h^{\mathrm{fr}}=\mathbf{\Gamma}_{q,\mathcal{A}}(\mathbf{\Gamma}_{\mathcal{A}\mathcal{A}}+\delta^{2}\mathbf{I})^{-1}\mathbf{y}$",
         "\\m{\\i{h\u0302}^{fr} = \\b{Γ}_{\\i{q},𝒜}(\\b{Γ}_{𝒜𝒜} + \\i{δ}^{2}\\b{I})^{−1}\\b{y}}"),
    ]
    for x, w, (title, formula, formula_ppt) in zip(xs, ws, steps):
        S.box(x, by, w, bh, "", C["phys"], C["phys_edge"])
        S.text(x + w / 2, by + 0.70 * bh, title, size=6.6)
        S.text(x + w / 2, by + 0.32 * bh, formula, size=7.3, ppt=formula_ppt)
    S.arrow((xs[0] + ws[0], by + bh / 2), (xs[1], by + bh / 2))
    S.arrow((xs[1] + ws[1], by + bh / 2), (xs[2], by + bh / 2))
    S.text(xs[0] + ws[0] / 2, by - 0.32, r"$\hat h^{\mathrm{diff}}=\hat X^{\mathrm{re}}+j\hat X^{\mathrm{im}}$",
           size=7.0, color=C["muted"],
           ppt="\\m{\\i{h\u0302}^{diff} = \\i{X\u0302}^{re} + \\i{j}\\i{X\u0302}^{im}}")
    x3c, x3r = xs[2] + ws[2] / 2, xs[2] + ws[2]
    S.text(x3c, by - 0.32, r"$\mathbf{y}$: measured complex CSI on $\mathcal{A}_{\mathrm{in}}$",
           size=7.0, color=C["muted"], ppt="\\m{\\b{y}}: measured complex CSI on \\m{𝒜_{in}}")

    # outputs of the physics prior
    ts = 1.3
    ox = 16.05
    S.thumb(d["physics"], ox, 10.55, ts, "jet", vmin, vmax, stack=2)
    S.thumb(d["logvar"], ox, 8.55, ts, "viridis", -1, 1)
    S.text(ox + ts + 0.35, 10.55 + ts / 2, r"Physics estimate" + "\n" + r"$\widehat{\mathbf{X}}^{\mathrm{fr}}$ (3 ch)",
           ha="left", ppt="Physics estimate\n\\m{\\b{X\u0302}^{fr}} (3 ch)")
    S.text(ox + ts + 0.35, 8.55 + ts / 2,
           "Confidence " + r"$\mathbf{V}$" + " (1 ch)" + "\nposterior variance " + E("eq:fr_variance"),
           ha="left", ppt="Confidence \\m{\\b{V}} (1 ch)\nposterior variance " + E("eq:fr_variance"))
    S.arrow((x3r, by + bh / 2 + 0.25), (ox - 0.05, 10.55 + ts / 2), cs="arc3,rad=-0.15")
    S.arrow((x3r, by + bh / 2 - 0.25), (ox - 0.05, 8.55 + ts / 2), cs="arc3,rad=0.15")

    # input routing into the physics panel
    S.line([xin + s + 0.18, 2.55], [9.75 + s / 2, 9.75 + s / 2])
    S.line([2.55, 2.55], [9.75 + s / 2, by + bh / 2])
    S.arrow((2.55, by + bh / 2), (xs[0], by + bh / 2))
    S.line([xin + s, 2.3], [6.75 + s / 2, 6.75 + s / 2])
    S.line([2.3, 2.3], [6.75 + s / 2, 8.75])
    S.line([2.3, x3c], [8.75, 8.75])
    S.arrow((x3c, 8.75), (x3c, by - 0.55))

    # ---------------- concatenation bar ----------------
    cx, cw, cy, chh = 2.85, 0.55, 0.85, 6.6
    S.box(cx, cy, cw, chh, "10-channel concatenation " + E("eq:pinn_input_tensor"), C["cbr"], C["cbr_edge"],
          size=7.6, rot=90, rounding=0.12)
    # diffusion estimate and mask to the bar
    S.line([2.55, 2.55], [9.75 + s / 2, 6.1])
    S.arrow((2.55, 6.1), (cx, 6.1))
    S.arrow((2.3, 5.3), (cx, 5.3))
    S.line([2.3, 2.3], [6.75 + s / 2, 5.3])
    S.line([xin + s + 0.18, 2.3], [1.05 + s / 2, 1.05 + s / 2])
    S.line([2.3, 2.3], [1.05 + s / 2, 2.0])
    S.arrow((2.3, 2.0), (cx, 2.0))
    # physics outputs down to the bar (routed along the panel bottom)
    S.line([ox + ts / 2, ox + ts / 2], [8.55, 7.75])
    S.line([3.65, ox + ts / 2], [7.75, 7.75])
    S.line([3.65, 3.65], [7.75, 4.4])
    S.arrow((3.65, 4.4), (cx + cw, 4.4))
    S.text(4.3, 7.78, r"$\widehat{\mathbf{X}}^{\mathrm{fr}}$, $\mathbf{V}$", size=6.8, color=C["muted"], ha="left",
           va="bottom", ppt="\\m{\\b{X\u0302}^{fr}}, \\m{\\b{V}}")

    # ---------------- refinement backbone ----------------
    col_w = 0.62
    blocks = [
        # x, w, y, h, text, fill, channels
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
        S.box(x, y, w, h, text, C[key], C[key + "_edge"], size=7.0, rot=90, rounding=0.12,
              ppt=mathtext_to_plain(text))
        S.text(x + w / 2, y + h + 0.2, str(ch), size=7.8, va="bottom", bold=True)
    mid = 4.0
    S.arrow((cx + cw, mid), (4.35, mid))
    for (x0, w0, *_), (x1, *_r) in zip(blocks[:-1], blocks[1:]):
        if x1 - (x0 + w0) > 0.4:
            S.arrow((x0 + w0, mid), (x1, mid))
    # group frames
    for x0, x1, label in [(4.2, 6.02, "Encoder"), (7.7, 9.52, "Bottleneck"), (12.1, 13.92, "Decoder")]:
        S.rect(x0, 0.95, x1 - x0, 6.15, C["ink"], dashed=True)
        S.text((x0 + x1) / 2, 0.72, label, size=7.8, va="top")
    S.text(6.86, 1.62, "Down\nsample", size=7.2, va="top")
    S.text(10.36 + 0.45, 1.62, "Up sample,\nproject", size=7.2, va="top")
    # skip connection
    S.line([5.56, 5.56], [1.6, 0.25], color=C["skip"])
    S.line([5.56, 11.85], [0.25, 0.25], color=C["skip"])
    S.arrow((11.85, 0.25), (11.85, mid - 0.05), color=C["skip"])
    S.text(8.7, 0.1, "Skip connection", va="top", color=C["skip"])

    # residual addition on the physics estimate
    ax_c = (15.75, mid)
    S.arrow((14.45 + col_w, mid), (ax_c[0] - 0.22, mid))
    S.circle(ax_c, 0.22, "+")
    S.text(15.2, mid + 0.18, r"$\Delta$", size=7.6, va="bottom", ppt="\\m{Δ}")
    S.line([ox + ts + 0.25, 17.6], [10.55 + ts / 2 - 0.25, 10.55 + ts / 2 - 0.25], color=C["muted"])
    S.line([17.6, 17.6], [10.55 + ts / 2 - 0.25, 7.45], color=C["muted"])
    S.line([15.75, 17.6], [7.45, 7.45], color=C["muted"])
    S.arrow((15.75, 7.45), (15.75, mid + 0.22), color=C["muted"])
    S.text(15.85, 6.6, r"$\widehat{\mathbf{X}}^{\mathrm{fr}}$", ha="left", color=C["muted"],
           ppt="\\m{\\b{X\u0302}^{fr}}")
    S.box(16.45, 1.9, col_w, 4.2,
          r"clip $[-1,1]$ " + E("eq:pinn_output") + "\nrestore measured " + E("eq:final_estimator"),
          C["out"], C["out_edge"], size=7.0, rot=90, rounding=0.12,
          ppt="clip [−1, 1] " + E("eq:pinn_output") + "\nrestore measured " + E("eq:final_estimator"))
    S.arrow((ax_c[0] + 0.22, mid), (16.45, mid))
    S.box(17.45, 1.9, col_w, 4.2, "Inverse normalization " + E("eq:inverse_normalization"), C["norm"],
          C["norm_edge"], size=7.0, rot=90, rounding=0.12)
    S.arrow((16.45 + col_w, mid), (17.45, mid))
    S.thumb(d["refined"], 18.75, mid - 0.78, 1.55, "jet", vmin, vmax, stack=2)
    S.arrow((17.45 + col_w, mid), (18.75, mid))
    S.text(18.75 + 0.78, mid - 1.0, "Refined CSI map", size=7.6, va="top")

    fig.savefig(OUT_DIR / "pinn_refiner_architecture.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(REPO / "runs" / "pinn_refiner_architecture_preview.png", dpi=160, bbox_inches="tight")
    print("saved", OUT_DIR / "pinn_refiner_architecture.pdf")
    to_pptx(S, OUT_DIR / "pinn_refiner_architecture.pptx")
    print("saved", OUT_DIR / "pinn_refiner_architecture.pptx")


if __name__ == "__main__":
    main()
