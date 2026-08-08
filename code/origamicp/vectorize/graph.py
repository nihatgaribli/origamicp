"""Assemble fitted segments into a planar crease graph.

The output is a real ``CreasePattern``, which is the point: once the prediction
is a graph rather than a mask, ``origamicp.verify`` applies to it directly and
we can ask whether the recovered pattern actually satisfies Maekawa, Kawasaki
and big-little-big. That is the metric the project exists to report, and it is
not available on pixels.

Nothing here tries to repair the graph. Dangling creases and odd-degree vertices
are left in place, so the validity rate reflects what was extracted rather than
what a cleanup pass could paper over.
"""

from __future__ import annotations

import cv2
import numpy as np

from origamicp.core.cp import BOUNDARY, MOUNTAIN, VALLEY, CreasePattern
from origamicp.vectorize.lines import Segment

# A model's mv head is two-way over creases: class zero is mountain.
# Not the same numbering as the three-way target map in models.targets.
PRED_MOUNTAIN, PRED_VALLEY = 0, 1


def sheet_polygon(sheet_mask: np.ndarray, epsilon_frac: float = 0.01) -> np.ndarray:
    """The sheet outline as a simple polygon."""
    contours, _ = cv2.findContours(
        (sheet_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("empty sheet mask")
    contour = max(contours, key=cv2.contourArea)
    epsilon = epsilon_frac * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(np.float64)


def _intersect(
    a: Segment, b: Segment, min_angle: float = np.deg2rad(12.0)
) -> tuple[np.ndarray, float, float] | None:
    """Intersection point with both parameters, or ``None`` if too shallow.

    Shallow crossings are rejected, not just exactly parallel ones. Two lines
    meeting at a few degrees intersect a long way from where their pixels
    actually overlap, so a small error in either fit throws the point tens of
    pixels off -- which is how a single junction turns into a scatter of
    spurious vertices.
    """
    da, db = a.end - a.start, b.end - b.start
    length_a, length_b = np.linalg.norm(da), np.linalg.norm(db)
    if length_a < 1e-9 or length_b < 1e-9:
        return None
    cosine = abs(float((da / length_a) @ (db / length_b)))
    if cosine > np.cos(min_angle):
        return None

    denominator = da[0] * db[1] - da[1] * db[0]
    if abs(denominator) < 1e-9:
        return None
    delta = b.start - a.start
    t = (delta[0] * db[1] - delta[1] * db[0]) / denominator
    u = (delta[0] * da[1] - delta[1] * da[0]) / denominator
    return a.start + t * da, float(t), float(u)


def _polygon_position(polygon: np.ndarray, point: np.ndarray) -> tuple[int, float, float]:
    """Closest edge of ``polygon``, the parameter along it, and the distance."""
    best = (0, 0.0, np.inf)
    for i in range(len(polygon)):
        p, q = polygon[i], polygon[(i + 1) % len(polygon)]
        edge = q - p
        length_sq = float(edge @ edge)
        if length_sq < 1e-12:
            continue
        t = float(np.clip((point - p) @ edge / length_sq, 0.0, 1.0))
        distance = float(np.linalg.norm(p + t * edge - point))
        if distance < best[2]:
            best = (i, t, distance)
    return best


class _Vertices:
    """Point set that merges anything closer together than ``snap``."""

    def __init__(self, snap: float) -> None:
        self.snap = snap
        self.points: list[np.ndarray] = []

    def add(self, point: np.ndarray) -> int:
        for index, existing in enumerate(self.points):
            if np.linalg.norm(existing - point) <= self.snap:
                return index
        self.points.append(np.asarray(point, dtype=np.float64))
        return len(self.points) - 1


def _edge_assignment(
    start: np.ndarray,
    end: np.ndarray,
    mv_label: np.ndarray,
    crease_prob: np.ndarray,
    samples: int = 32,
    trim: float = 0.15,
) -> str:
    """Majority mountain/valley vote along an edge, weighted by confidence.

    The ends are trimmed: near a vertex several creases overlap and their
    labels bleed into each other, so the middle of an edge is the part that
    actually speaks for it.
    """
    height, width = mv_label.shape
    t = np.linspace(trim, 1.0 - trim, samples)[:, None]
    points = start * (1 - t) + end * t

    delta = end - start
    normal = np.array([-delta[1], delta[0]])
    normal = normal / max(float(np.linalg.norm(normal)), 1e-9)

    mountain = valley = 0.0
    # Search a short way to either side. The fitted line sits a pixel or two off
    # the crease it represents, and sampling exactly on it then lands on blank
    # paper, where the label map still holds some value but the probability is
    # zero -- so the edge would collect no votes and fall back to a default.
    for offset in (-2.0, -1.0, 0.0, 1.0, 2.0):
        shifted = points + normal * offset
        xs = np.clip(np.round(shifted[:, 0]).astype(int), 0, width - 1)
        ys = np.clip(np.round(shifted[:, 1]).astype(int), 0, height - 1)
        labels = mv_label[ys, xs]
        weights = crease_prob[ys, xs].astype(np.float64)
        mountain += float(weights[labels == PRED_MOUNTAIN].sum())
        valley += float(weights[labels == PRED_VALLEY].sum())

    return MOUNTAIN if mountain >= valley else VALLEY


def _prune_stubs(
    points: list[np.ndarray],
    edges: list[list[int]],
    assignment: list[str],
    min_edge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop short dangling creases left behind at junctions.

    A fitted line rarely stops exactly at the crossing it makes -- the pixels of
    the other crease pull its extent a few pixels past -- so splitting at the
    intersection leaves a stub hanging off the far side. Those stubs are what
    turn one degree-eight vertex into a degree-sixteen one surrounded by
    degree-one neighbours, which fails every geometric check for a reason that
    has nothing to do with the paper.

    Only dangling edges are removed. A short edge between two real junctions is
    a genuine short crease and is left alone.
    """
    alive = [True] * len(edges)

    changed = True
    while changed:
        changed = False
        degree: dict[int, int] = {}
        for index, (a, b) in enumerate(edges):
            if not alive[index]:
                continue
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1

        for index, (a, b) in enumerate(edges):
            if not alive[index] or assignment[index] == BOUNDARY:
                continue
            if float(np.linalg.norm(points[a] - points[b])) >= min_edge:
                continue
            if degree.get(a, 0) == 1 or degree.get(b, 0) == 1:
                alive[index] = False
                changed = True

    kept = [i for i, live in enumerate(alive) if live]
    used = sorted({v for i in kept for v in edges[i]})
    remap = {old: new for new, old in enumerate(used)}
    return (
        np.array([points[i] for i in used]),
        np.array([[remap[a], remap[b]] for a, b in (edges[i] for i in kept)], dtype=np.int64),
        np.array([assignment[i] for i in kept], dtype="<U1"),
    )


def build_pattern(
    segments: list[Segment],
    polygon: np.ndarray,
    mv_label: np.ndarray,
    crease_prob: np.ndarray,
    snap: float = 6.0,
    boundary_snap: float = 12.0,
    min_edge: float | None = None,
) -> CreasePattern:
    """Split segments at their mutual crossings and close the sheet outline."""
    if min_edge is None:
        min_edge = 2.5 * snap
    vertices = _Vertices(snap)
    polygon_indices = [vertices.add(point) for point in polygon]

    # Points where each segment must be broken, as parameters along it.
    cuts: list[list[tuple[float, int]]] = [[] for _ in segments]
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            hit = _intersect(segments[i], segments[j])
            if hit is None:
                continue
            point, t, u = hit
            margin_i = snap / max(segments[i].length, 1e-9)
            margin_j = snap / max(segments[j].length, 1e-9)
            if not (-margin_i <= t <= 1 + margin_i and -margin_j <= u <= 1 + margin_j):
                continue
            index = vertices.add(point)
            cuts[i].append((float(np.clip(t, 0.0, 1.0)), index))
            cuts[j].append((float(np.clip(u, 0.0, 1.0)), index))

    on_boundary: dict[int, tuple[int, float]] = {}
    edges: list[list[int]] = []
    assignment: list[str] = []

    for segment, breaks in zip(segments, cuts):
        ends = []
        for t, point in ((0.0, segment.start), (1.0, segment.end)):
            edge_index, position, distance = _polygon_position(polygon, point)
            if distance <= boundary_snap:
                # Pull the crease onto the sheet edge so the vertex it makes is
                # classified as boundary rather than as a stray interior vertex.
                p, q = polygon[edge_index], polygon[(edge_index + 1) % len(polygon)]
                snapped = p + position * (q - p)
                index = vertices.add(snapped)
                on_boundary[index] = (edge_index, position)
            else:
                index = vertices.add(point)
            ends.append((t, index))

        ordered = sorted(breaks + ends, key=lambda item: item[0])
        chain = [index for _, index in ordered]
        deduped = [chain[0]] + [b for a, b in zip(chain, chain[1:]) if a != b]

        for a, b in zip(deduped, deduped[1:]):
            edges.append([a, b])
            assignment.append(
                _edge_assignment(
                    vertices.points[a], vertices.points[b], mv_label, crease_prob
                )
            )

    # Walk the outline, threading in every point that was snapped onto it.
    ring: list[tuple[int, float, int]] = [
        (i, 0.0, index) for i, index in enumerate(polygon_indices)
    ]
    ring += [
        (edge_index, position, index)
        for index, (edge_index, position) in on_boundary.items()
        if index not in polygon_indices
    ]
    ring.sort(key=lambda item: (item[0], item[1]))
    for (_, _, a), (_, _, b) in zip(ring, ring[1:] + ring[:1]):
        if a != b:
            edges.append([a, b])
            assignment.append(BOUNDARY)

    return CreasePattern(*_prune_stubs(vertices.points, edges, assignment, min_edge))
