"""Decoding a crease graph under the fold theorems.

The extractor produces each crease independently: a line fitted to its own
pixels, a label voted from its own neighbourhood. The paper does not work that
way. At every interior vertex the sector angles must satisfy Kawasaki's
condition and the mountain/valley labels must satisfy Maekawa's, and those are
joint constraints over all the creases meeting there. This module puts the
independent estimates back under those constraints.

A warning about how to evaluate it, because the obvious way is circular.
Enforcing Kawasaki and then reporting Kawasaki satisfaction measures nothing --
it is one by construction. The same goes for Maekawa. Constrained decoding is a
denoising step, so it has to be judged against ground truth: does the corrected
assignment agree with the true one more often, and do the corrected angles sit
closer to the true angles? Those are the questions
``scripts/constrained_decode.py`` asks. The geometric validity rate remains
useful, but only as a diagnostic of the *raw* output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from origamicp.core.cp import MOUNTAIN, VALLEY, CreasePattern
from origamicp.verify.single_vertex import valid_assignments


def project_angles_kawasaki(angles: np.ndarray) -> np.ndarray:
    """Nearest sector angles that sum to 2pi and alternate to zero.

    Both conditions are linear, so the closest point satisfying them has a
    closed form. For even degree the two constraint rows are orthogonal and of
    equal norm, which collapses the projection to one subtraction per angle.
    """
    angles = np.asarray(angles, dtype=np.float64)
    k = len(angles)
    if k == 0 or k % 2:
        return angles.copy()

    signs = (-1.0) ** np.arange(k)
    total_error = float(angles.sum()) - 2.0 * np.pi
    alternating_error = float(signs @ angles)
    return angles - (total_error + signs * alternating_error) / k


def solve_vertex_mv(
    angles: np.ndarray, scores: np.ndarray, fixed: dict[int, str] | None = None
) -> list[str] | None:
    """Highest-scoring foldable assignment for one vertex.

    ``scores[i]`` is the evidence that crease ``i`` is a mountain, positive for
    mountain and negative for valley. Candidates come from the exact crimp
    solver, so every one of them genuinely folds flat -- Maekawa, the
    big-little-big lemma and the crimp reduction are all satisfied, not just the
    first of them.

    ``fixed`` pins creases already decided at a neighbouring vertex. Returns
    ``None`` when nothing consistent with those exists.
    """
    options = valid_assignments(np.asarray(angles, dtype=np.float64))
    if not options:
        return None

    best, best_score = None, -np.inf
    for option in options:
        if fixed and any(option[i] != kind for i, kind in fixed.items()):
            continue
        score = sum(
            scores[i] if letter == MOUNTAIN else -scores[i]
            for i, letter in enumerate(option)
        )
        if score > best_score:
            best, best_score = option, score
    return list(best) if best is not None else None


@dataclass
class DecodeReport:
    vertices: int = 0
    relabelled: int = 0  # creases whose label the constraints changed
    unsatisfiable: int = 0  # vertices with no foldable assignment at all
    conflicts: int = 0  # vertices that had to overrule an already-fixed crease

    def summary(self) -> str:
        return (
            f"{self.vertices} interior vertices | {self.relabelled} creases relabelled | "
            f"{self.unsatisfiable} unsatisfiable | {self.conflicts} conflicts"
        )


def constrain_pattern(
    cp: CreasePattern,
    edge_scores: np.ndarray,
    angle_projection: bool = True,
) -> tuple[CreasePattern, DecodeReport]:
    """Re-decode MV labels, and optionally correct the vertex geometry.

    Vertices are visited most-confident first and their creases then held fixed,
    so a vertex the model was sure about is not overturned by a neighbour it was
    guessing at. A crease shared by two vertices can still be contested; when
    the second vertex has no assignment consistent with the first, it takes its
    best unconstrained one and the disagreement is counted rather than hidden.
    """
    assignment = np.array(cp.assignment, copy=True)
    scores = np.asarray(edge_scores, dtype=np.float64)
    report = DecodeReport()

    interior = cp.interior_vertices()
    confidence = []
    for v in interior:
        edges, _ = cp.sorted_creases(v)
        confidence.append(float(np.mean(np.abs(scores[edges]))) if len(edges) else 0.0)
    order = [interior[i] for i in np.argsort(confidence)[::-1]]

    decided: dict[int, str] = {}
    for v in order:
        edges, _ = cp.sorted_creases(v)
        angles = cp.sector_angles(v)
        if len(edges) < 4 or len(edges) % 2:
            continue
        report.vertices += 1

        if angle_projection:
            angles = project_angles_kawasaki(angles)

        pinned = {i: decided[e] for i, e in enumerate(edges) if e in decided}
        solved = solve_vertex_mv(angles, scores[edges], pinned)
        if solved is None:
            solved = solve_vertex_mv(angles, scores[edges])
            if solved is None:
                report.unsatisfiable += 1
                continue
            if pinned:
                report.conflicts += 1

        for i, e in enumerate(edges):
            if assignment[e] != solved[i]:
                report.relabelled += 1
            assignment[e] = solved[i]
            decided[e] = solved[i]

    vertices = cp.vertices
    if angle_projection:
        vertices = _reposition_for_kawasaki(cp)

    return CreasePattern(vertices, cp.edges, assignment), report


def _reposition_for_kawasaki(cp: CreasePattern) -> np.ndarray:
    """Rotate each crease so its vertex satisfies Kawasaki exactly.

    Only the far endpoint of a crease that dangles from exactly one interior
    vertex is moved, and it slides along its own radius. Endpoints shared by two
    interior vertices are left alone: pulling on both ends would let one
    vertex's correction undo the other's, and there is no reason to trust either
    over the other.
    """
    vertices = np.array(cp.vertices, dtype=np.float64, copy=True)
    interior = set(cp.interior_vertices())

    for v in sorted(interior):
        edges, directions = cp.sorted_creases(v)
        angles = cp.sector_angles(v)
        if len(edges) < 4 or len(edges) % 2:
            continue

        corrected = project_angles_kawasaki(angles)
        # Rebuild the crease directions from the corrected sectors, anchored on
        # the first crease so the whole fan is not rotated for no reason.
        offsets = np.concatenate([[0.0], np.cumsum(corrected)[:-1]])
        new_directions = directions[0] + offsets

        for edge, direction in zip(edges, new_directions):
            other = cp.other_end(int(edge), v)
            if other in interior:
                continue
            radius = float(np.linalg.norm(cp.vertices[other] - cp.vertices[v]))
            vertices[other] = vertices[v] + radius * np.array(
                [np.cos(direction), np.sin(direction)]
            )
    return vertices
