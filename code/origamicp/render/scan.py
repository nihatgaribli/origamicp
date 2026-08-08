"""Render a crease pattern as a scan of physically folded paper.

Why this is not a line drawing. The whole task rests on one cue: a mountain
under raking light is bright on the side facing the lamp and dark on the far
side, and a valley is the other way round. Drawing red and blue lines would
hand the model a label channel that does not exist on real paper, and the
transfer to real scans would fail completely.

So we render the physics instead. Creases become a signed height field -- ridge
for a mountain, trough for a valley -- and the image is that surface shaded by a
low light. The mountain/valley cue then arises the same way it does on paper,
which is the only version of it a model can carry over to the real dataset.

Consequence worth remembering when reading results: flip ``light_azimuth_deg``
by 180 degrees and every crease's appearance inverts. Light direction is part of
the label, not a nuisance, and the manifest records it for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from origamicp.core.cp import BOUNDARY, MOUNTAIN, CreasePattern

SCANNER, PHOTOMETRIC, BACKLIT = "scanner", "photometric", "backlit"


@dataclass
class ScanStyle:
    """Capture and material parameters. Defaults imitate a 600 dpi flatbed scan."""

    size_px: int = 768
    mode: str = SCANNER
    light_azimuth_deg: float = 45.0
    light_elevation_deg: float = 18.0

    # These three set the difficulty of the whole dataset and are the only
    # parameters that must be calibrated against real scans -- see
    # ``measure_crease_snr``, which puts these defaults at a cue roughly three
    # times the paper's own texture. Comfortably readable, which is what a
    # default should be; the dataset generator samples a range around it so the
    # benchmark spans easy to genuinely hard.
    crease_height: float = 0.90  # ridge amplitude; this is the signal
    undulation: float = 0.45  # unfolded paper never lies perfectly flat
    fiber_noise: float = 0.035

    crease_width_px: float = 1.6
    crease_wobble_px: float = 0.6  # hand-folded creases are not perfectly straight
    crease_strength_jitter: float = 0.35
    albedo_variation: float = 0.03
    ambient: float = 0.55
    sensor_noise: float = 0.004
    # Flat paper must land below the ceiling. Render it at 1.0 and the lit side
    # of every crease clips to white, which erases exactly half the MV cue --
    # valleys survive on their shadow while mountains lose their highlight.
    # Real scans of white paper sit around 0.8-0.9 anyway.
    paper_level: float = 0.82

    margin_frac: float = 0.07
    background: int = 26  # the black card the protocol puts behind the sheet
    perspective: float = 0.0  # 0 for a flatbed, ~0.02 for a handheld photo
    rotation_deg: float | None = None  # None means "sample it"


def _sheet_corners(style: ScanStyle, rng: np.random.Generator) -> np.ndarray:
    size = style.size_px
    centre = np.array([size / 2, size / 2])
    angle = np.deg2rad(
        rng.uniform(0, 360) if style.rotation_deg is None else style.rotation_deg
    )

    # A square rotated by theta needs |cos| + |sin| times its half-side to fit,
    # so the side has to shrink with the angle. Sizing it as if it were axis
    # aligned pushes the corners out of frame, and a clipped sheet no longer has
    # four findable corners -- registration then falls back to the bounding
    # rectangle and reports every orientation as equally good.
    extent = abs(np.cos(angle)) + abs(np.sin(angle))
    available = size / 2 - style.margin_frac * size - style.perspective * size * 3
    half = max(available, size * 0.05) / extent

    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64) * half
    corners = square @ rot.T + centre

    if style.perspective > 0:
        corners = corners + rng.normal(0, style.perspective * size, corners.shape)
    return corners


def _project(cp: CreasePattern, corners: np.ndarray) -> np.ndarray:
    lo, hi = cp.vertices.min(axis=0), cp.vertices.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    unit = (cp.vertices - lo) / span
    # Paper y points up, image rows go down.
    src = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src, corners.astype(np.float32))
    return cv2.perspectiveTransform(unit[None].astype(np.float64), homography)[0]


def _height_field(
    cp: CreasePattern, pixels: np.ndarray, style: ScanStyle, rng: np.random.Generator
) -> np.ndarray:
    """Signed relief: positive where the paper ridges toward the camera."""
    size = style.size_px
    field = np.zeros((size, size), dtype=np.float32)

    for (a, b), kind in zip(cp.edges, cp.assignment):
        if kind == BOUNDARY:
            continue  # the sheet edge is a cut, not a fold
        p, q = pixels[a], pixels[b]
        sign = 1.0 if kind == MOUNTAIN else -1.0
        strength = sign * rng.uniform(
            1 - style.crease_strength_jitter, 1 + style.crease_strength_jitter
        )

        n_steps = max(2, int(np.linalg.norm(q - p) / 24))
        t = np.linspace(0, 1, n_steps + 1)[:, None]
        points = p * (1 - t) + q * t
        if style.crease_wobble_px > 0 and n_steps > 1:
            direction = q - p
            normal = np.array([-direction[1], direction[0]])
            normal = normal / max(np.linalg.norm(normal), 1e-9)
            wobble = rng.normal(0, style.crease_wobble_px, size=len(points))
            wobble[0] = wobble[-1] = 0.0  # endpoints are pinned to the sheet edge
            points = points + normal[None, :] * wobble[:, None]

        layer = np.zeros_like(field)
        cv2.polylines(layer, [np.round(points).astype(np.int32)], False, 1.0, 1, cv2.LINE_AA)
        field += layer * strength

    # Blurring a one-pixel line gives the crease its cross-sectional profile.
    field = cv2.GaussianBlur(field, (0, 0), style.crease_width_px) * style.crease_height

    if style.undulation > 0:
        coarse = rng.normal(0, 1, (size // 16 + 2, size // 16 + 2)).astype(np.float32)
        coarse = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)
        field += _unit_std(cv2.GaussianBlur(coarse, (0, 0), size / 22)) * style.undulation

    if style.fiber_noise > 0:
        fibers = rng.normal(0, 1, (size, size)).astype(np.float32)
        field += _unit_std(cv2.GaussianBlur(fibers, (0, 0), 0.7)) * style.fiber_noise

    return field


def _unit_std(array: np.ndarray) -> np.ndarray:
    """Rescale to unit standard deviation.

    Blurring collapses the variance of white noise by an amount that depends on
    the kernel, so without this the style parameters would not mean what they
    say and retuning one would silently rescale the others.
    """
    return array / max(float(array.std()), 1e-9)


def _shade(field: np.ndarray, style: ScanStyle) -> np.ndarray:
    """Lambertian shading under a single low light."""
    d_row, d_col = np.gradient(field)
    normals = np.stack([-d_col, -d_row, np.ones_like(field)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    azimuth = np.deg2rad(style.light_azimuth_deg)
    elevation = np.deg2rad(style.light_elevation_deg)
    light = np.array(
        [
            np.cos(azimuth) * np.cos(elevation),
            np.sin(azimuth) * np.cos(elevation),
            np.sin(elevation),
        ]
    )
    lambert = normals @ light
    # Divide out the grazing angle so flat paper reads mid-grey rather than
    # black, leaving the creases as the only modulation.
    return lambert / max(np.sin(elevation), 1e-6)


def render(
    cp: CreasePattern, style: ScanStyle | None = None, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Render ``cp``; returns ``(image_bgr, sheet_corners_in_pixels)``.

    The corners come back so the caller can register against them directly
    instead of re-detecting, which keeps synthetic ground truth exact.
    """
    style = style or ScanStyle()
    rng = rng or np.random.default_rng()

    corners = _sheet_corners(style, rng)
    pixels = _project(cp, corners)
    field = _height_field(cp, pixels, style, rng)

    if style.mode == BACKLIT:
        # Transmission: compressed fibres pass more light, so both fold
        # directions brighten equally and MV is genuinely unrecoverable.
        creases = np.abs(field - cv2.GaussianBlur(field, (0, 0), 6.0))
        image = style.paper_level * (0.55 + 2.6 * creases)
    else:
        # _shade returns 1.0 on flat paper, so this sits at paper_level and the
        # creases modulate either side of it.
        image = style.paper_level * (
            style.ambient + (1.0 - style.ambient) * _shade(field, style)
        )

    texture = cv2.GaussianBlur(
        rng.normal(0, 1, field.shape).astype(np.float32), (0, 0), 12.0
    )
    image = image * (1.0 + _unit_std(texture) * style.albedo_variation)

    if style.mode == PHOTOMETRIC:  # handheld shots fall off toward the corners
        ys, xs = np.mgrid[0 : style.size_px, 0 : style.size_px].astype(np.float32)
        radius = np.hypot(xs - style.size_px / 2, ys - style.size_px / 2)
        image *= 1.0 - 0.22 * (radius / radius.max()) ** 2

    image += rng.normal(0, style.sensor_noise, image.shape)

    # Mask to the pattern's own outline, not to the four corners: a sheet with a
    # chamfered corner is not a rectangle, and the cut has to show in the image
    # or it cannot serve as a fiducial.
    boundary = cp.boundary_polygon()
    outline = pixels[boundary] if len(boundary) >= 3 else corners
    mask = np.zeros(field.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(outline).astype(np.int32)], 255)
    out = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    out = np.where(mask > 0, out, np.uint8(style.background))

    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR), corners


def photometric_stack(
    cp: CreasePattern,
    azimuths_deg=(0, 45, 90, 135, 180, 225, 270, 315),
    style: ScanStyle | None = None,
    seed: int = 0,
) -> list[tuple[float, np.ndarray]]:
    """The same sheet under a rotating light, as the capture protocol prescribes.

    Every frame is drawn from the same seed so the paper, the creases and the
    pose are identical and only the lighting moves -- which is exactly what a
    locked-down camera on a tripod gives you, and what photometric stereo needs.
    """
    frames = []
    for azimuth in azimuths_deg:
        variant = ScanStyle(**{**vars(style or ScanStyle()), "mode": PHOTOMETRIC})
        variant.light_azimuth_deg = float(azimuth)
        variant.rotation_deg = variant.rotation_deg if variant.rotation_deg is not None else 0.0
        image, _ = render(cp, variant, np.random.default_rng(seed))
        frames.append((float(azimuth), image))
    return frames


def measure_crease_snr(
    cp: CreasePattern,
    style: ScanStyle | None = None,
    rng: np.random.Generator | None = None,
    offset_px: float = 3.5,
) -> tuple[float, float]:
    """How readable the mountain/valley cue is, against the paper's own texture.

    Returns ``(signal, noise)`` in grey levels. ``signal`` is the mean brightness
    step across a crease -- the thing that says mountain or valley. ``noise`` is
    the same measurement taken on blank paper, so the ratio answers the only
    question that matters: can the cue be seen at all?

    This is the handle for matching the renders to reality. Measure it on the
    real pilot scans, then set ``crease_height``, ``undulation`` and
    ``fiber_noise`` so the synthetic ratio lands in the same place. Without that
    calibration the synthetic set is either impossible or trivial, and both fail
    to transfer.
    """
    style = style or ScanStyle()
    rng = rng or np.random.default_rng(0)

    image, corners = render(cp, style, rng)
    return crease_snr_of_image(image, cp, corners, rng, offset_px)


def crease_snr_of_image(
    image: np.ndarray,
    cp: CreasePattern,
    corners: np.ndarray,
    rng: np.random.Generator | None = None,
    offset_px: float = 3.5,
) -> tuple[float, float]:
    """The same measurement on an image that already exists.

    Split out so difficulty can be measured on the dataset as generated, rather
    than on a fresh render with the same parameters. The two differ by the
    random draw, and binning results by a difficulty the images do not actually
    have would blur every curve.
    """
    rng = rng or np.random.default_rng(0)
    gray = (
        image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ).astype(np.float64)
    pixels = _project(cp, corners)

    def step(point: np.ndarray, normal: np.ndarray) -> float | None:
        near = point + normal * offset_px
        far = point - normal * offset_px
        h, w = gray.shape
        for p in (near, far):
            if not (2 <= p[0] < w - 2 and 2 <= p[1] < h - 2):
                return None
        return float(
            gray[int(near[1]), int(near[0])] - gray[int(far[1]), int(far[0])]
        )

    signals = []
    for (a, b), kind in zip(cp.edges, cp.assignment):
        if kind == BOUNDARY:
            continue
        p, q = pixels[a], pixels[b]
        direction = q - p
        length = np.linalg.norm(direction)
        if length < 8 * offset_px:
            continue
        normal = np.array([-direction[1], direction[0]]) / length
        steps = [
            s
            for t in np.linspace(0.2, 0.8, 15)
            if (s := step(p + direction * t, normal)) is not None
        ]
        if steps:
            signals.append(abs(float(np.mean(steps))))

    # The background must be measured exactly as the signal is -- averaged along
    # a line of the same length. Comparing a 15-sample mean against single-pixel
    # noise understates the ratio by roughly sqrt(15) and would have us tuning
    # the renderer to fix a defect in the measurement.
    centre = corners.mean(axis=0)
    radius = 0.3 * np.linalg.norm(corners[0] - centre)
    span = 15 * offset_px
    background = []
    for _ in range(200):
        angle = rng.uniform(0, 2 * np.pi)
        point = centre + rng.uniform(0, radius) * np.array([np.cos(angle), np.sin(angle)])
        phi = rng.uniform(0, 2 * np.pi)
        along = np.array([np.cos(phi), np.sin(phi)])
        normal = np.array([-along[1], along[0]])
        steps = [
            s
            for t in np.linspace(-0.5, 0.5, 15)
            if (s := step(point + along * span * t, normal)) is not None
        ]
        if steps:
            background.append(float(np.mean(steps)))

    signal = float(np.mean(signals)) if signals else 0.0
    noise = float(np.std(background)) if background else 0.0
    return signal, noise
