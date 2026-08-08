"""Corner chamfer: the physical fiducial that pins a sheet's orientation.

A square sheet looks the same from four directions, so registration has to
guess among them and settle ties with the crease pattern itself. That works
until the pattern is symmetric. A Miura sheet turned 180 degrees maps onto
itself while every mountain becomes a valley -- geometrically identical, wholly
mislabelled, and nothing downstream would notice.

Cutting one corner off breaks the symmetry once and for all, for every design,
at the cost of one snip per sheet. The chamfer lives in the crease pattern's
boundary rather than in metadata, so ``back_face`` mirrors it, ``to_svg`` prints
it as a cut line and the renderer masks it, all without special cases.

Convention: the chamfer sits at the front-view origin corner -- lowest x, then
lowest y. On the printed template, which is the back face, it appears mirrored.
Cut where the template shows it and the convention takes care of itself.
"""

from __future__ import annotations

import numpy as np

from origamicp.core.cp import BOUNDARY, CreasePattern

DEFAULT_CUT_MM = 8.0


def _corner_vertex(cp: CreasePattern, corner: np.ndarray) -> int:
    boundary = cp.boundary_polygon()
    if len(boundary) == 0:
        raise ValueError("pattern has no boundary")
    distances = np.linalg.norm(cp.vertices[boundary] - corner, axis=1)
    return int(boundary[int(np.argmin(distances))])


def chamfer(
    cp: CreasePattern, cut: float = DEFAULT_CUT_MM, corner: str = "min"
) -> CreasePattern:
    """Cut ``cut`` millimetres off one corner of the sheet.

    ``corner`` selects which bounding-box corner: "min" (default), "maxx",
    "maxy" or "max".
    """
    lo, hi = cp.vertices.min(axis=0), cp.vertices.max(axis=0)
    target = {
        "min": np.array([lo[0], lo[1]]),
        "maxx": np.array([hi[0], lo[1]]),
        "maxy": np.array([lo[0], hi[1]]),
        "max": np.array([hi[0], hi[1]]),
    }[corner]

    v = _corner_vertex(cp, target)
    incident = cp.incident_edges(v)
    if any(cp.assignment[e] != BOUNDARY for e in incident):
        raise ValueError(f"vertex {v} carries a crease; cannot chamfer there")
    if len(incident) != 2:
        raise ValueError(f"corner vertex {v} has {len(incident)} boundary edges, expected 2")

    origin = cp.vertices[v]
    vertices = list(cp.vertices)
    edges = [list(map(int, e)) for e in cp.edges]
    assignment = list(cp.assignment)

    replacements = []
    for edge in incident:
        other = cp.other_end(edge, v)
        direction = cp.vertices[other] - origin
        length = float(np.linalg.norm(direction))
        if length <= cut:
            raise ValueError(f"cut of {cut} exceeds the {length:.1f} edge at the corner")
        new_index = len(vertices)
        vertices.append(origin + direction / length * cut)
        # Re-attach the boundary edge to the new vertex instead of the corner.
        edges[edge] = [new_index, other]
        replacements.append(new_index)

    edges.append(replacements)
    assignment.append(BOUNDARY)

    # The old corner vertex is now unused; drop it so the boundary stays a
    # simple cycle and vertex counts keep matching the geometry.
    keep = [i for i in range(len(vertices)) if i != v]
    remap = {old: new for new, old in enumerate(keep)}
    return CreasePattern(
        np.array([vertices[i] for i in keep]),
        np.array([[remap[a], remap[b]] for a, b in edges]),
        np.array(assignment, dtype="<U1"),
    )


def is_chamfered(cp: CreasePattern, tol: float = 1e-6) -> bool:
    """True when one bounding-box corner has been cut away."""
    return chamfered_corner(cp, tol) is not None


def chamfered_corner(cp: CreasePattern, tol: float = 1e-6) -> np.ndarray | None:
    """The bounding-box corner that was cut off, or ``None``.

    Found by asking which corner no vertex sits on any more -- robust to the
    pattern having been rotated, mirrored or rescaled since the cut was made.
    """
    lo, hi = cp.vertices.min(axis=0), cp.vertices.max(axis=0)
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    distances = np.linalg.norm(
        cp.vertices[None, :, :] - corners[:, None, :], axis=2
    ).min(axis=1)

    missing = np.argsort(distances)[::-1]
    far, next_far = distances[missing[0]], distances[missing[1]]
    if far <= tol or far < 2.0 * max(next_far, tol):
        return None  # every corner occupied, or no corner clearly emptier
    return corners[missing[0]]
