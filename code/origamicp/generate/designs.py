"""The pilot design set: a complexity ladder from one crease to a full tessellation.

Every pattern here is built in millimetres on a square sheet and is checked by
``origamicp.verify`` in the test suite, so a template that cannot fold flat can
never reach the printer.

The single-vertex designs are deliberately placed off-centre. A pattern with
four-fold symmetry has no recoverable orientation -- registration would report
it as ambiguous and every sheet would need manual checking. Moving the vertex
costs nothing mathematically (the fold theorems constrain angles, not position)
and makes the whole pilot align automatically.
"""

from __future__ import annotations

import numpy as np

from origamicp.core.cp import BOUNDARY, MOUNTAIN, VALLEY, CreasePattern

SHEET_MM = 150.0


def _perimeter_t(point: np.ndarray, size: float) -> float:
    """Position of a boundary point along the square's perimeter, in [0, 4*size)."""
    x, y = point
    if np.isclose(y, 0.0):
        return x
    if np.isclose(x, size):
        return size + y
    if np.isclose(y, size):
        return 3 * size - x
    return 4 * size - y


def _ray_to_boundary(origin: np.ndarray, angle_deg: float, size: float) -> np.ndarray:
    """Where a ray leaving ``origin`` meets the square [0, size]^2."""
    direction = np.array([np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))])
    best = np.inf
    for axis in (0, 1):
        for bound in (0.0, size):
            if abs(direction[axis]) < 1e-12:
                continue
            t = (bound - origin[axis]) / direction[axis]
            if t > 1e-9:
                hit = origin + t * direction
                other = hit[1 - axis]
                if -1e-9 <= other <= size + 1e-9 and t < best:
                    best = t
    if not np.isfinite(best):
        raise ValueError(f"ray at {angle_deg} deg never leaves the sheet")
    return origin + best * direction


def single_vertex_sheet(
    sector_angles_deg,
    mv: str,
    centre=(0.42, 0.56),
    first_crease_deg: float = 12.0,
    size: float = SHEET_MM,
) -> CreasePattern:
    """One interior vertex on a square sheet, creases running out to the edges.

    ``centre`` is given as a fraction of the sheet so designs stay comparable
    across sheet sizes. The boundary is split wherever a crease meets it.
    """
    sectors = np.asarray(sector_angles_deg, dtype=np.float64)
    if not np.isclose(sectors.sum(), 360.0):
        raise ValueError(f"sector angles sum to {sectors.sum()}, expected 360")
    if len(mv) != len(sectors):
        raise ValueError(f"{len(sectors)} sectors but {len(mv)} MV labels")

    origin = np.array(centre, dtype=np.float64) * size
    directions = first_crease_deg + np.concatenate([[0.0], np.cumsum(sectors)[:-1]])
    hits = [_ray_to_boundary(origin, d, size) for d in directions]

    vertices = [origin] + hits
    edges = [[0, i + 1] for i in range(len(hits))]
    assignment = list(mv)

    corners = [
        np.array([0.0, 0.0]),
        np.array([size, 0.0]),
        np.array([size, size]),
        np.array([0.0, size]),
    ]
    ring = [(_perimeter_t(h, size), i + 1) for i, h in enumerate(hits)]
    for corner in corners:
        ring.append((_perimeter_t(corner, size), len(vertices)))
        vertices.append(corner)
    ring.sort()

    for k in range(len(ring)):
        edges.append([ring[k][1], ring[(k + 1) % len(ring)][1]])
        assignment.append(BOUNDARY)

    return CreasePattern(np.array(vertices), np.array(edges), np.array(assignment, dtype="<U1"))


def pleat(n_creases: int = 8, size: float = SHEET_MM, offsets=None) -> CreasePattern:
    """A plain accordion: parallel creases, no interior vertices at all.

    Rung zero of the ladder. With nothing for the fold theorems to constrain it
    isolates pure crease *detection* from mountain/valley reasoning.

    ``offsets`` allows uneven panel widths. Alternating M/V still folds flat
    with unequal panels -- the sheet just zigzags at varying pitch -- whereas an
    arbitrary assignment on unequal panels can force the paper through itself,
    so the alternation is not negotiable here.
    """
    vertices = [
        np.array([0.0, 0.0]),
        np.array([size, 0.0]),
        np.array([size, size]),
        np.array([0.0, size]),
    ]
    edges, assignment = [], []

    xs = np.linspace(0, size, n_creases + 2)[1:-1] if offsets is None else np.asarray(offsets, dtype=float)
    lower, upper = [(0.0, 0), (size, 1)], [(0.0, 3), (size, 2)]
    for k, x in enumerate(xs):
        bottom, top = len(vertices), len(vertices) + 1
        vertices += [np.array([x, 0.0]), np.array([x, size])]
        edges.append([bottom, top])
        assignment.append(MOUNTAIN if k % 2 == 0 else VALLEY)
        lower.append((x, bottom))
        upper.append((x, top))

    for row in (lower, upper):
        row.sort()
        for a, b in zip(row, row[1:]):
            edges.append([a[1], b[1]])
            assignment.append(BOUNDARY)
    for pair in ([0, 3], [1, 2]):
        edges.append(pair)
        assignment.append(BOUNDARY)

    return CreasePattern(np.array(vertices), np.array(edges), np.array(assignment, dtype="<U1"))


def miura(
    n_cols: int = 4,
    n_rows: int = 4,
    zigzag: float = 0.22,
    size: float = SHEET_MM,
    row_heights=None,
    row_zigzags=None,
) -> CreasePattern:
    """Miura-ori on a square sheet.

    The top and bottom rows are flattened so the sheet is a true square rather
    than a zigzag-edged strip. That is safe: a vertex's sector angles depend on
    where its neighbours sit, and the vertical neighbours stay directly above
    and below regardless of the row's offset, so no interior vertex is affected.

    MV rule: vertical creases alternate by row, zigzag creases are uniform. That
    puts one vertical M and one vertical V at every interior vertex, which with
    the two zigzag creases gives Maekawa's required three-to-one split.

    ``row_heights`` and ``row_zigzags`` allow an irregular corrugation. Rows may
    vary freely, but *columns* may not: Kawasaki at a vertex reduces to the two
    zigzag rays having equal slope, which holds only while the column spacing is
    uniform. Row spacing never enters that condition.
    """
    a = size / n_cols

    if row_heights is None:
        row_heights = np.full(n_rows, size / n_rows)
    row_heights = np.asarray(row_heights, dtype=float) * (size / np.sum(row_heights))
    y_base = np.concatenate([[0.0], np.cumsum(row_heights)])

    if row_zigzags is None:
        row_zigzags = zigzag * row_heights
    row_zigzags = np.asarray(row_zigzags, dtype=float)

    index = {}
    vertices = []
    for i in range(n_rows + 1):
        for j in range(n_cols + 1):
            y = y_base[i]
            if 0 < i < n_rows:
                y += (j % 2) * row_zigzags[i]
            index[(i, j)] = len(vertices)
            vertices.append(np.array([j * a, y]))

    edges, assignment = [], []
    for i in range(n_rows):
        for j in range(n_cols + 1):
            on_edge = j in (0, n_cols)
            edges.append([index[(i, j)], index[(i + 1, j)]])
            assignment.append(BOUNDARY if on_edge else (MOUNTAIN if i % 2 == 0 else VALLEY))
    for i in range(n_rows + 1):
        for j in range(n_cols):
            on_edge = i in (0, n_rows)
            edges.append([index[(i, j)], index[(i, j + 1)]])
            assignment.append(BOUNDARY if on_edge else MOUNTAIN)

    return CreasePattern(np.array(vertices), np.array(edges), np.array(assignment, dtype="<U1"))


PILOT_DESIGNS = {
    "d01_pleat": lambda: pleat(8),
    "d02_vertex4": lambda: single_vertex_sheet([60, 120, 120, 60], "MMMV"),
    "d03_vertex6": lambda: single_vertex_sheet([80, 40, 70, 100, 30, 40], "MMVMMV"),
    "d04_vertex8": lambda: single_vertex_sheet(
        [30, 55, 70, 50, 45, 35, 35, 40], "MVMMVMMV"
    ),
    "d05_miura_small": lambda: miura(4, 4),
    "d06_miura_large": lambda: miura(8, 8),
}
