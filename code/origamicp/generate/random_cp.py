"""Random crease patterns for synthetic pre-training.

Three hundred real sheets is not enough to train a crease extractor from
scratch, so the plan is to pre-train on synthetic renders and fine-tune on
paper. That only helps if the synthetic patterns are genuinely foldable --
otherwise the network learns the statistics of impossible paper.

So nothing here is sampled and hoped for. Single-vertex patterns are checked by
the exact crimp test; corrugations and pleats come from families that are
foldable by construction. Every generated pattern is valid ground truth.
"""

from __future__ import annotations

import numpy as np

from origamicp.core.cp import CreasePattern
from origamicp.generate.designs import SHEET_MM, miura, pleat, single_vertex_sheet
from origamicp.verify.single_vertex import valid_assignments

MIN_SECTOR_DEG = 16.0

# Creases closer together than this are excluded, for two independent reasons
# that happen to agree. Physically, a hand folder cannot place two parallel
# creases a couple of millimetres apart on a 15 cm sheet with any accuracy.
# Practically, the capture protocol scans at 600 dpi but training renders are far
# coarser, and below roughly this spacing the line fitter merges neighbours into
# one -- measured recall drops from 0.90 to 0.52 across that boundary. Generating
# patterns the pipeline cannot express only manufactures unreachable error.
MIN_CREASE_GAP_MM = 8.0


def _spaced(fractions: np.ndarray, size: float, minimum: float) -> bool:
    """True when consecutive positions are at least ``minimum`` apart."""
    positions = np.concatenate([[0.0], np.cumsum(fractions) * size])
    return bool(np.all(np.diff(positions) >= minimum))


def random_sector_angles(
    rng: np.random.Generator, degree: int, min_angle_deg: float = MIN_SECTOR_DEG
) -> np.ndarray:
    """Sector angles in degrees satisfying Kawasaki's condition.

    Kawasaki says the alternating sum vanishes, which for an even degree is the
    same as the odd-indexed and even-indexed sectors each summing to 180. So we
    sample two halves independently rather than sampling freely and rejecting
    almost everything.
    """
    if degree % 2 or degree < 4:
        raise ValueError(f"degree must be even and at least 4, got {degree}")
    half = degree // 2
    if min_angle_deg * half >= 180.0:
        raise ValueError(f"degree {degree} cannot fit sectors of {min_angle_deg} deg")

    angles = np.empty(degree)
    for parity in (0, 1):
        while True:
            part = rng.dirichlet(np.full(half, 2.5)) * 180.0
            if part.min() >= min_angle_deg:
                break
        angles[parity::2] = part
    return angles


def random_single_vertex(
    rng: np.random.Generator, degree: int | None = None, size: float = SHEET_MM
) -> CreasePattern:
    """One interior vertex, placed off-centre so its orientation is recoverable."""
    for _ in range(64):
        deg = degree if degree is not None else int(rng.choice([4, 4, 6, 6, 8, 10]))
        angles = random_sector_angles(rng, deg)
        options = valid_assignments(np.deg2rad(angles))
        if not options:
            continue  # rare, but not every geometry admits an assignment
        return single_vertex_sheet(
            angles,
            options[rng.integers(len(options))],
            centre=(rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)),
            first_crease_deg=rng.uniform(0.0, 360.0),
            size=size,
        )
    raise RuntimeError("could not sample a foldable vertex")


def random_pleat(rng: np.random.Generator, size: float = SHEET_MM) -> CreasePattern:
    """An accordion with uneven panel widths."""
    limit = max(2, int(size / MIN_CREASE_GAP_MM) - 1)
    n = int(rng.integers(3, min(15, limit) + 1))
    for _ in range(200):
        widths = rng.dirichlet(np.full(n + 1, 4.0))
        if _spaced(widths, size, MIN_CREASE_GAP_MM):
            break
    else:  # fall back to even spacing rather than emit an unfoldable pattern
        widths = np.full(n + 1, 1.0 / (n + 1))
    return pleat(n, size=size, offsets=np.cumsum(widths)[:-1] * size)


def random_corrugation(rng: np.random.Generator, size: float = SHEET_MM) -> CreasePattern:
    """Miura-ori with irregular row heights and zigzag depths."""
    limit = max(2, int(size / MIN_CREASE_GAP_MM))
    n_cols = int(rng.integers(2, min(9, limit) + 1))
    n_rows = int(rng.integers(2, min(9, limit) + 1))
    for _ in range(200):
        heights = rng.dirichlet(np.full(n_rows, 6.0))
        if _spaced(heights, size, MIN_CREASE_GAP_MM):
            break
    else:
        heights = np.full(n_rows, 1.0 / n_rows)
    zigzags = heights * size * rng.uniform(0.10, 0.40, size=n_rows)
    return miura(
        n_cols=n_cols,
        n_rows=n_rows,
        size=size,
        row_heights=heights,
        row_zigzags=np.concatenate([[0.0], zigzags[1:]]),
    )


FAMILIES = {
    "single_vertex": random_single_vertex,
    "pleat": random_pleat,
    "corrugation": random_corrugation,
}
FAMILY_WEIGHTS = {"single_vertex": 0.45, "pleat": 0.20, "corrugation": 0.35}


def random_design(
    rng: np.random.Generator, size: float = SHEET_MM
) -> tuple[str, CreasePattern]:
    """Sample a family, then a pattern from it. Returns ``(family, pattern)``."""
    names = list(FAMILIES)
    weights = np.array([FAMILY_WEIGHTS[n] for n in names])
    family = names[int(rng.choice(len(names), p=weights / weights.sum()))]
    return family, FAMILIES[family](rng, size=size)
