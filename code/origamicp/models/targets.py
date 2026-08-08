"""Pixel-level targets: turn a crease pattern plus a pose into training masks.

Two masks, because the task has two halves of very different difficulty.
Finding a crease is largely an edge-detection problem; deciding whether it is a
mountain or a valley depends on a shading asymmetry a few grey levels wide.
Supervising them separately keeps the easy half from dominating the loss and
lets each be reported on its own.

One consequence of the physics is worth stating plainly, because it constrains
the whole model design. Under a single light, a mountain lit from the left is
pixel-for-pixel a valley lit from the right. A single image therefore fixes the
mountain/valley assignment only up to one global flip, and the flip can only be
resolved by knowing where the light was -- Maekawa's condition cannot do it,
since inverting every crease leaves the defect at plus or minus two. That is why
the capture manifest records the light azimuth and why the model is given it.
"""

from __future__ import annotations

import cv2
import numpy as np

from origamicp.core.cp import BOUNDARY, MOUNTAIN, CreasePattern

# Two different label conventions exist in this codebase and confusing them
# is silent: the target map is three-way with background at zero, while a
# model's argmax is two-way over creases only. They are named apart so the
# mismatch cannot compile.
TARGET_BACKGROUND, TARGET_MOUNTAIN, TARGET_VALLEY = 0, 1, 2
DEFAULT_THICKNESS = 3


def project_to_pixels(cp: CreasePattern, corners: np.ndarray) -> np.ndarray:
    """Vertex coordinates in image pixels, given the sheet's four corners.

    Mirrors the renderer's own mapping so targets land exactly where the creases
    were drawn.
    """
    lo, hi = cp.vertices.min(axis=0), cp.vertices.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    unit = (cp.vertices - lo) / span
    src = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src, corners.astype(np.float32))
    return cv2.perspectiveTransform(unit[None].astype(np.float64), homography)[0]


def build_targets(
    cp: CreasePattern,
    corners: np.ndarray,
    size_px: int,
    thickness: int = DEFAULT_THICKNESS,
) -> dict[str, np.ndarray]:
    """Crease mask, MV labels and sheet mask for one rendered sheet.

    ``crease``  uint8, 1 on a fold line.
    ``mv``      uint8, 0 off-crease, 1 mountain, 2 valley.
    ``sheet``   uint8, 1 inside the paper -- loss is only meaningful there.

    Creases are drawn a few pixels wide rather than one: a hairline target makes
    the objective nearly all background and punishes a prediction that is right
    but half a pixel over.
    """
    pixels = project_to_pixels(cp, corners)
    crease = np.zeros((size_px, size_px), dtype=np.uint8)
    mv = np.zeros((size_px, size_px), dtype=np.uint8)

    for (a, b), kind in zip(cp.edges, cp.assignment):
        if kind == BOUNDARY:
            continue  # the sheet edge is a cut, not a fold
        p = tuple(np.round(pixels[a]).astype(int))
        q = tuple(np.round(pixels[b]).astype(int))
        cv2.line(crease, p, q, 1, thickness, cv2.LINE_8)
        label = TARGET_MOUNTAIN if kind == MOUNTAIN else TARGET_VALLEY
        cv2.line(mv, p, q, label, thickness, cv2.LINE_8)

    sheet = np.zeros((size_px, size_px), dtype=np.uint8)
    boundary = cp.boundary_polygon()
    outline = pixels[boundary] if len(boundary) >= 3 else corners
    cv2.fillPoly(sheet, [np.round(outline).astype(np.int32)], 1)

    # A crease drawn thick can spill past a chamfered corner; clip it back.
    crease &= sheet
    mv *= sheet

    return {"crease": crease, "mv": mv, "sheet": sheet}


def light_channels(
    azimuth_deg: float | None, size_px: int, elevation_deg: float | None = None
) -> np.ndarray:
    """Two constant planes encoding where the light came from.

    ``None`` means no directional light -- a back-lit scan -- and gives zeros,
    which is the honest encoding: those images carry no MV information at all.
    """
    if azimuth_deg is None:
        return np.zeros((2, size_px, size_px), dtype=np.float32)
    angle = np.deg2rad(azimuth_deg)
    scale = 1.0 if elevation_deg is None else float(np.cos(np.deg2rad(elevation_deg)))
    planes = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * scale
    return np.repeat(planes[:, None, None], size_px, axis=1).repeat(size_px, axis=2)
