"""Native (editable) PowerPoint equations from LaTeX, for python-pptx text boxes.

LaTeX -> MathML (latex2mathml) -> Office Math / OMML (Microsoft's MML2OMML.XSL, shipped with Office)
-> ``a14:m`` math zones inside the text box. The shape is wrapped in ``mc:AlternateContent`` with a
picture fallback (the label rendered by LaTeX), as PowerPoint does itself, so viewers without
Office Math support (LibreOffice, older Office) show the LaTeX rendering instead.

Labels use matplotlib syntax: text with ``$...$`` math, ``\n`` = new paragraph.
"""

import copy
import os
import re

from lxml import etree

MML2OMML = os.environ.get("MML2OMML_XSL", r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL")
NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "a14": "http://schemas.microsoft.com/office/drawing/2010/main",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "mml": "http://www.w3.org/1998/Math/MathML",
}
_XSLT = None


def q(tag):
    pre, name = tag.split(":")
    return f"{{{NS[pre]}}}{name}"


def _xslt():
    global _XSLT
    if _XSLT is None:
        if not os.path.exists(MML2OMML):
            raise SystemExit(f"MML2OMML.XSL not found at {MML2OMML}; set MML2OMML_XSL to its location.")
        _XSLT = etree.XSLT(etree.parse(MML2OMML))
    return _XSLT


def latex_to_omml(tex):
    """LaTeX math (without $) -> m:oMath element."""
    from latex2mathml.converter import convert

    mml = etree.fromstring(convert(tex).encode("utf-8"))
    for mover in mml.iter(q("mml:mover")):  # \widehat / \widetilde: mark as accents, not limits
        mover.set("accent", "true")
    for mi in mml.iter(q("mml:mi")):
        t = mi.text or ""
        if len(t) == 1 and 0x1D400 <= ord(t) <= 0x1D433:
            # math-bold letters (\mathbf{d}) -> plain letter + bold style; PowerPoint drops the
            # astral-plane character when it shares a run with other characters
            k = ord(t) - 0x1D400
            mi.text = chr(ord("A") + k) if k < 26 else chr(ord("a") + k - 26)
            mi.set("mathvariant", "bold")
        elif len(t) == 1 and 0x0391 <= ord(t) <= 0x03A9 and mi.get("mathvariant") is None:
            mi.set("mathvariant", "normal")  # capital Greek is upright, as in LaTeX
    omath = _xslt()(mml).getroot()
    etree.cleanup_namespaces(omath, top_nsmap={"m": NS["m"]})
    return omath


def _rpr(size_pt, color, bold=False, italic=False, math=False):
    rpr = etree.Element(q("a:rPr"), lang="en-US", sz=str(int(round(size_pt * 100))), b="1" if bold else "0",
                        i="1" if italic else "0", dirty="0")
    fill = etree.SubElement(rpr, q("a:solidFill"))
    etree.SubElement(fill, q("a:srgbClr"), val=color.lstrip("#").upper())
    face = "Cambria Math" if math else "Arial"
    etree.SubElement(rpr, q("a:latin"), typeface=face)
    etree.SubElement(rpr, q("a:cs"), typeface=face)
    return rpr


def _style_math_runs(omath, size_pt, color):
    """PowerPoint math runs carry DrawingML run properties (size, color, font) after m:rPr."""
    for r in omath.iter(q("m:r")):
        sty = r.find(f"{q('m:rPr')}/{q('m:sty')}")
        val = sty.get(q("m:val")) if sty is not None else "i"
        rpr = _rpr(size_pt, color, bold=val in ("b", "bi"), italic=val in ("i", "bi"), math=True)
        mrpr = r.find(q("m:rPr"))
        r.insert(0 if mrpr is None else 1, rpr)


def fill_paragraphs(txbody, label, size_pt, color, bold=False, align="ctr"):
    """Write a matplotlib-style label (text + $math$) into an a:txBody as runs and math zones."""
    for p in txbody.findall(q("a:p")):
        txbody.remove(p)
    for line in label.split("\n"):
        p = etree.SubElement(txbody, q("a:p"))
        etree.SubElement(p, q("a:pPr"), algn=align)
        for k, seg in enumerate(re.split(r"\$", line)):
            if not seg:
                continue
            if k % 2:  # math
                zone = etree.SubElement(p, q("a14:m"))
                omath = latex_to_omml(seg)
                _style_math_runs(omath, size_pt, color)
                zone.append(omath)
            else:
                r = etree.SubElement(p, q("a:r"))
                r.append(_rpr(size_pt, color, bold=bold))
                t = etree.SubElement(r, q("a:t"))
                t.text = seg
        p.append(_rpr(size_pt, color, bold=bold))
        p[-1].tag = q("a:endParaRPr")


def wrap_with_fallback(sp, fallback_pic):
    """Put ``sp`` (uses a14 math) in mc:AlternateContent with ``fallback_pic`` (a p:pic) as fallback."""
    parent = sp.getparent()
    idx = parent.index(sp)
    alt = etree.Element(q("mc:AlternateContent"), nsmap={"mc": NS["mc"]})
    choice = etree.SubElement(alt, q("mc:Choice"), nsmap={"a14": NS["a14"]}, Requires="a14")
    fallback = etree.SubElement(alt, q("mc:Fallback"))
    parent.remove(sp)
    fallback_pic.getparent().remove(fallback_pic)
    choice.append(sp)
    fallback.append(fallback_pic)
    parent.insert(idx, alt)
    return alt


def has_math(label):
    return "$" in label


__all__ = ["fill_paragraphs", "wrap_with_fallback", "latex_to_omml", "has_math", "copy"]
