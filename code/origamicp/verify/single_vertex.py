"""Exact flat-foldability for a single vertex, by recursive crimping.

Single-vertex patterns are the one tier where we can decide foldability outright
-- the general problem is NP-hard, but one vertex is polynomial. That makes them
the anchor of the benchmark: exact ground truth, no solver, no approximation.

The algorithm is the standard crimp reduction. A sector strictly smaller than
both its neighbours has to tuck between them, so the two creases bounding it
must differ; removing that pair leaves a smaller vertex with the same answer.
Repeat until two creases remain.

Where this goes beyond the textbook statement: when several sectors tie for
smallest, the reduction is no longer forced, so we branch over every admissible
crimp instead of committing to one. That keeps the search exhaustive rather
than relying on the tie-free assumption. Degrees stay small enough (<= 12 in
practice) that the branching costs nothing.

The local conditions in ``local.py`` remain the right metric for *extracted*
patterns -- they are per-vertex, cheap and differentiable-ish. This module is
for generating data we can promise is foldable.
"""

from __future__ import annotations

import numpy as np

from origamicp.core.cp import MOUNTAIN, VALLEY

DEFAULT_TOL = np.deg2rad(1e-6)


def _crimp(angles: np.ndarray, mv: np.ndarray, sector: int):
    """Remove the two creases bounding ``sector`` and merge the three sectors.

    Returns ``None`` when the merge would produce a non-positive sector, which
    means the paper would have to pass through itself.
    """
    k = len(angles)
    shift = (sector - 1) % k
    a = np.roll(angles, -shift)
    m = np.roll(mv, -shift)

    merged = a[0] - a[1] + a[2]
    if merged <= DEFAULT_TOL:
        return None
    return np.concatenate([[merged], a[3:]]), np.concatenate([[m[0]], m[3:]])


def is_flat_foldable(angles: np.ndarray, mv: np.ndarray, tol: float = DEFAULT_TOL) -> bool:
    """Decide whether one vertex folds flat with the given MV assignment.

    ``angles[i]`` is the sector between crease ``i`` and crease ``i+1``, matching
    ``CreasePattern.sector_angles`` / ``mv_around``.
    """
    angles = np.asarray(angles, dtype=np.float64)
    mv = np.asarray(mv, dtype="<U1")
    k = len(angles)

    if k != len(mv) or k % 2 or k < 2:
        return False
    if not np.all(np.isin(mv, [MOUNTAIN, VALLEY])):
        return False
    if abs(((-1.0) ** np.arange(k)) @ angles) > tol * max(k, 1):
        return False  # Kawasaki is preserved by crimping, so check it once

    if k == 2:
        # Two collinear creases are the same physical fold: same direction.
        return mv[0] == mv[1]

    for i in range(k):
        if angles[i] > angles[i - 1] + tol or angles[i] > angles[(i + 1) % k] + tol:
            continue  # not a local minimum, strict or tied
        if mv[i] == mv[(i + 1) % k]:
            continue  # the little sector cannot tuck between two like folds
        reduced = _crimp(angles, mv, i)
        if reduced is not None and is_flat_foldable(*reduced, tol=tol):
            return True
    return False


def valid_assignments(angles: np.ndarray, limit: int | None = None) -> list[str]:
    """Every MV assignment that folds the given sector angles flat.

    Enumerates the assignments satisfying Maekawa (the only ones that can work)
    and keeps those the crimp test accepts.
    """
    from itertools import combinations

    angles = np.asarray(angles, dtype=np.float64)
    k = len(angles)
    if k % 2:
        return []

    found = []
    for n_mountains in ((k + 2) // 2, (k - 2) // 2):
        for positions in combinations(range(k), n_mountains):
            mv = np.array([VALLEY] * k, dtype="<U1")
            mv[list(positions)] = MOUNTAIN
            if is_flat_foldable(angles, mv):
                found.append("".join(mv))
                if limit is not None and len(found) >= limit:
                    return found
    return found
