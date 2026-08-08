"""Hand-buildable crease patterns used for tests and as generator building blocks.

Single-vertex patterns are the base case of the whole benchmark: they are the
only case where flat-foldability is decidable in polynomial time, so they give
us exact ground truth to calibrate everything else against.
"""

from __future__ import annotations

import numpy as np

from origamicp.core.cp import BOUNDARY, CreasePattern


def single_vertex(
    sector_angles_deg, mv: str | list[str], radius: float = 1.0
) -> CreasePattern:
    """A star pattern: one interior vertex with the requested sector angles.

    ``sector_angles_deg[i]`` is the angle between crease ``i`` and crease
    ``i+1`` (cyclic) and must sum to 360. ``mv[i]`` assigns crease ``i``.

    The outer endpoints are joined by boundary edges so that exactly one vertex
    is interior, which is what the local conditions apply to.
    """
    sectors = np.asarray(sector_angles_deg, dtype=np.float64)
    mv = list(mv)
    if len(mv) != len(sectors):
        raise ValueError(f"{len(sectors)} sectors but {len(mv)} MV labels")
    if not np.isclose(sectors.sum(), 360.0):
        raise ValueError(f"sector angles sum to {sectors.sum()}, expected 360")

    k = len(sectors)
    # Crease i points at the cumulative angle of the sectors before it.
    directions = np.deg2rad(np.concatenate([[0.0], np.cumsum(sectors)[:-1]]))

    center = np.zeros((1, 2))
    outer = radius * np.stack([np.cos(directions), np.sin(directions)], axis=1)
    vertices = np.vstack([center, outer])

    creases = [[0, i + 1] for i in range(k)]
    border = [[i + 1, (i + 1) % k + 1] for i in range(k)]
    edges = np.array(creases + border, dtype=np.int64)
    assignment = np.array(mv + [BOUNDARY] * k, dtype="<U1")

    return CreasePattern(vertices, edges, assignment)


def unit_square_boundary() -> tuple[np.ndarray, list[list[int]], list[str]]:
    """The four corners and boundary edges of a unit square sheet."""
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    return vertices, edges, [BOUNDARY] * 4
