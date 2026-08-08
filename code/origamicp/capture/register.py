"""Align a digital crease pattern to a scan of the physical sheet.

This is what makes ground truth free. We already know the exact ``.fold`` the
sheet was folded from, so we never annotate creases by hand -- we only have to
find the homography that maps paper coordinates onto image pixels.

A square sheet has no distinguishing landmark, so its four corners leave the
correspondence ambiguous up to four rotations and a mirror. Rather than trust
the operator to have recorded the orientation correctly, we try all eight
candidates and keep whichever projects the pattern onto actual crease pixels.
The runner-up's score comes back as a confidence margin: an ambiguous sheet
should be looked at, not silently trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from origamicp.core.cp import BOUNDARY, MOUNTAIN, VALLEY, CreasePattern

_SAMPLES_PER_EDGE = 48


def ridge_response(
    image: np.ndarray, sigma: float = 6.0, presmooth: float = 1.5
) -> np.ndarray:
    """Highlight crease pixels: a band-pass magnitude, normalised to [0, 1].

    A crease under raking light is a paired highlight and shadow, so we take the
    absolute deviation from the local mean. That responds to mountains and
    valleys alike, which is what we want for *alignment* -- telling the two
    apart is the model's job, not the registrar's.

    ``presmooth`` is what makes it usable on paper. Paper fibre gives a plain
    high-pass almost as much energy as the creases do, and the orientation
    search then scores every candidate alike. Smoothing to just below crease
    width first suppresses the fibre and leaves the line-scale structure.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0
    if presmooth > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), presmooth)
    band_pass = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), sigma))
    hi = np.percentile(band_pass, 99.5)
    if hi <= 0:
        return np.zeros_like(band_pass)
    return np.clip(band_pass / hi, 0.0, 1.0)


def _order_ccw(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    centre = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    return points[np.argsort(angles)]


@dataclass
class SheetOutline:
    corners: np.ndarray  # four corners, counter-clockwise
    chamfer_index: int | None  # which corner was cut off, if any


def _line_intersection(p1, p2, p3, p4) -> np.ndarray | None:
    d1, d2 = p2 - p1, p4 - p3
    denominator = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denominator) < 1e-9:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denominator
    return p1 + t * d1


def detect_sheet_outline(image: np.ndarray) -> SheetOutline:
    """Find the sheet's corners, and the chamfer if one was cut.

    Assumes a dark backing behind the paper: scanning white paper against the
    scanner's white lid gives almost no edge to find, so the capture protocol
    calls for a sheet of black card on top.

    A chamfered sheet traces a pentagon whose shortest side is the cut. The
    missing corner is recovered by extending its two neighbours until they meet,
    which keeps the four-corner homography unchanged while recording which
    corner the cut identifies.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("no sheet found; check that a dark backing was used")
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)

    for frac in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(contour, frac * perimeter, True).reshape(-1, 2)

        if len(approx) == 5:
            lengths = [
                np.linalg.norm(approx[(i + 1) % 5] - approx[i]) for i in range(5)
            ]
            k = int(np.argmin(lengths))
            # A genuine chamfer is clearly the shortest side; otherwise the
            # pentagon is just a rounded or dented corner and we ignore it.
            if lengths[k] < 0.5 * float(np.median(lengths)):
                corner = _line_intersection(
                    approx[(k - 1) % 5], approx[k],
                    approx[(k + 1) % 5], approx[(k + 2) % 5],
                )
                if corner is not None:
                    quad = np.array(
                        [corner, *(approx[(k + 2 + i) % 5] for i in range(3))]
                    )
                    ordered = _order_ccw(quad)
                    index = int(np.argmin(np.linalg.norm(ordered - corner, axis=1)))
                    return SheetOutline(ordered, index)

        if len(approx) == 4:
            return SheetOutline(_order_ccw(approx), None)

    # Rounded or slightly torn corners never reduce to four points; the
    # minimum-area rectangle is a good enough fallback for a flattened sheet.
    return SheetOutline(_order_ccw(cv2.boxPoints(cv2.minAreaRect(contour))), None)


def detect_sheet_quad(image: np.ndarray) -> np.ndarray:
    """The sheet's four corners, counter-clockwise."""
    return detect_sheet_outline(image).corners


def _cp_corners(cp: CreasePattern) -> np.ndarray:
    lo, hi = cp.vertices.min(axis=0), cp.vertices.max(axis=0)
    return _order_ccw(
        np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    )


def _crease_points(cp: CreasePattern, homography: np.ndarray) -> np.ndarray:
    """Sample points along every non-boundary crease, projected into the image."""
    interior = [i for i in range(cp.n_edges) if cp.assignment[i] != BOUNDARY]
    if not interior:
        return np.zeros((0, 2))

    t = np.linspace(0.0, 1.0, _SAMPLES_PER_EDGE)[:, None]
    starts = cp.vertices[cp.edges[interior, 0]]
    ends = cp.vertices[cp.edges[interior, 1]]
    pts = (starts[:, None, :] * (1 - t) + ends[:, None, :] * t).reshape(-1, 2)

    projected = cv2.perspectiveTransform(pts[None].astype(np.float64), homography)
    return projected[0]


BACKGROUND_FLOOR = 0.02


def _score(response: np.ndarray, points: np.ndarray, background: float) -> float:
    """Excess ridge response along the projected creases, relative to the sheet.

    Reported as a ratio above background rather than a raw mean. An absolute
    mean rewards any placement that lands on textured paper, so a wrong
    orientation scores nearly as well as the right one; dividing by the sheet's
    own median makes a miss score zero and separates the candidates cleanly.
    """
    if len(points) == 0:
        return 0.0
    h, w = response.shape
    xs = np.clip(np.round(points[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(points[:, 1]).astype(int), 0, h - 1)
    ratio = float(response[ys, xs].mean()) / max(background, BACKGROUND_FLOOR)
    return max(ratio - 1.0, 0.0)


def _background_level(response: np.ndarray, quad: np.ndarray) -> float:
    """Typical response inside the sheet. Median, so dense patterns do not
    inflate their own baseline the way a mean would."""
    mask = np.zeros(response.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
    inside = response[mask > 0]
    return float(np.median(inside)) if inside.size else 0.0


@dataclass
class RegistrationResult:
    homography: np.ndarray
    score: float
    margin: float  # best / runner-up; near 1.0 means the orientation is ambiguous
    rotation_k: int
    mirrored: bool
    quad: np.ndarray
    used_fiducial: bool = False

    @property
    def confident(self) -> bool:
        # A chamfer fixes the orientation geometrically, so there is no tie for
        # the crease score to break and the margin carries no information.
        if self.used_fiducial:
            return self.score > 0.15
        return self.score > 0.15 and self.margin > 1.15

    def project(self, cp: CreasePattern) -> np.ndarray:
        """Vertex coordinates of ``cp`` in image pixels."""
        return cv2.perspectiveTransform(
            cp.vertices[None].astype(np.float64), self.homography
        )[0]


def _fiducial_candidates(cp: CreasePattern, outline: SheetOutline):
    """The (mirror, roll) pairs that line the sheet's chamfer up with the pattern's.

    ``getPerspectiveTransform`` pairs ``src[i]`` with ``dst[i]``, and
    ``np.roll(a, k)[i] == a[(i - k) % 4]``, so requiring the two chamfered
    corners to correspond fixes ``k`` for each mirror state. Eight candidates
    become two, and the 180-degree ambiguity that would silently invert every
    label on a symmetric pattern simply cannot arise.
    """
    from origamicp.generate.fiducial import chamfered_corner

    if outline.chamfer_index is None:
        return None
    corner = chamfered_corner(cp)
    if corner is None:
        return None

    src = _cp_corners(cp)
    source_index = int(np.argmin(np.linalg.norm(src - corner, axis=1)))
    image_index = outline.chamfer_index
    return [
        (False, (source_index - image_index) % 4),
        (True, (source_index - 3 + image_index) % 4),
    ]


def register(
    image: np.ndarray, cp: CreasePattern, quad: np.ndarray | None = None
) -> RegistrationResult:
    """Fit the homography taking ``cp`` paper coordinates to ``image`` pixels."""
    outline = SheetOutline(quad, None) if quad is not None else detect_sheet_outline(image)
    quad = outline.corners
    allowed = _fiducial_candidates(cp, outline)
    response = ridge_response(image)
    # Tolerate a pixel or two of misalignment so the score reflects orientation,
    # not sub-pixel fit.
    dilated = cv2.dilate(response, np.ones((3, 3), np.float32))
    background = _background_level(dilated, quad)

    src = _cp_corners(cp).astype(np.float32)
    poses = allowed if allowed is not None else [
        (mirrored, k) for mirrored in (False, True) for k in range(4)
    ]

    candidates = []
    for mirrored, k in poses:
        ordered = quad[::-1] if mirrored else quad
        dst = np.roll(ordered, k, axis=0).astype(np.float32)
        homography = cv2.getPerspectiveTransform(src, dst)
        points = _crease_points(cp, homography)
        candidates.append((_score(dilated, points, background), k, mirrored, homography))

    candidates.sort(key=lambda c: -c[0])
    best, runner_up = candidates[0], candidates[1]
    margin = best[0] / runner_up[0] if runner_up[0] > 1e-9 else float("inf")

    return RegistrationResult(
        homography=best[3],
        score=best[0],
        margin=margin,
        rotation_k=best[1],
        mirrored=best[2],
        quad=quad,
        used_fiducial=allowed is not None,
    )


def overlay(
    image: np.ndarray, cp: CreasePattern, result: RegistrationResult, thickness: int = 2
) -> np.ndarray:
    """Draw the aligned pattern over the scan so a human can check it.

    Every registered sheet gets one of these. Reviewing a contact sheet of
    overlays takes minutes and catches the misalignments that no score does.
    """
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    colours = {MOUNTAIN: (0, 0, 255), VALLEY: (255, 0, 0), BOUNDARY: (0, 200, 0)}
    projected = result.project(cp)
    for (a, b), kind in zip(cp.edges, cp.assignment):
        cv2.line(
            canvas,
            tuple(np.round(projected[a]).astype(int)),
            tuple(np.round(projected[b]).astype(int)),
            colours.get(kind, (0, 255, 255)),
            thickness,
            cv2.LINE_AA,
        )

    label = f"score={result.score:.3f} margin={result.margin:.2f} k={result.rotation_k} mirror={result.mirrored}"
    cv2.putText(canvas, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    cv2.putText(canvas, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return canvas
