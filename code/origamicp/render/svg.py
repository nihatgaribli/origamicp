"""Printable SVG templates.

SVG is written by hand rather than through a plotting library because the one
thing that matters here is *physical* accuracy: the file declares its size in
millimetres, so printing at 100% scale puts the creases exactly where the
digital pattern says they are. Any resampling step would quietly break the
correspondence that makes our ground truth free.
"""

from __future__ import annotations

from pathlib import Path

from origamicp.core.cp import BOUNDARY, MOUNTAIN, VALLEY, CreasePattern

# Origami convention: mountains dash-dot, valleys dashed. Colour is redundant
# with the dash pattern on purpose, so a greyscale printer still works.
STYLES = {
    MOUNTAIN: 'stroke="#c1272d" stroke-width="0.45" stroke-dasharray="5,1.2,1,1.2"',
    VALLEY: 'stroke="#1b5e9c" stroke-width="0.45" stroke-dasharray="3,2"',
    BOUNDARY: 'stroke="#9a9a9a" stroke-width="0.35"',
}
DEFAULT_STYLE = 'stroke="#000000" stroke-width="0.3"'

MARGIN_MM = 10.0
LABEL_MM = 8.0


def to_svg(
    cp: CreasePattern,
    path: str | Path,
    label: str = "",
    sheet_mm: float = 150.0,
) -> Path:
    """Write a fold-along template sized in millimetres.

    The pattern is assumed to already live in millimetre coordinates spanning
    ``sheet_mm``. The label sits outside the sheet outline so it is removed when
    the square is cut out and never reaches the scanner.
    """
    width = sheet_mm + 2 * MARGIN_MM
    height = width + LABEL_MM

    def x(v: float) -> float:
        return v + MARGIN_MM

    def y(v: float) -> float:
        return sheet_mm - v + MARGIN_MM  # SVG's y axis points down

    lines = []
    for (a, b), kind in zip(cp.edges, cp.assignment):
        p, q = cp.vertices[a], cp.vertices[b]
        style = STYLES.get(kind, DEFAULT_STYLE)
        lines.append(
            f'  <line x1="{x(p[0]):.4f}" y1="{y(p[1]):.4f}" '
            f'x2="{x(q[0]):.4f}" y2="{y(q[1]):.4f}" {style} stroke-linecap="round"/>'
        )

    caption = (
        f'  <text x="{MARGIN_MM:.2f}" y="{height - 2.5:.2f}" font-family="sans-serif" '
        f'font-size="4" fill="#666">{label}</text>'
        if label
        else ""
    )

    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}mm" height="{height}mm" '
            f'viewBox="0 0 {width} {height}">',
            '  <rect width="100%" height="100%" fill="#ffffff"/>',
            *lines,
            caption,
            "</svg>",
            "",
        ]
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return path
