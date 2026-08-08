"""Crease-level matching between an extracted pattern and its ground truth.

Pixel F1 answers a question nobody asked. A crease recovered one pixel to the
side is a success for the task and a partial failure for a pixel metric, while
a scatter of blobs on blank paper costs a pixel metric little and would wreck a
graph. Matching whole creases -- same line, sufficient overlap -- measures what
the downstream folding actually depends on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from origamicp.core.cp import BOUNDARY, CreasePattern


def crease_segments(cp: CreasePattern, pixels: np.ndarray | None = None) -> list[np.ndarray]:
    """Non-boundary edges as ``(2, 2)`` endpoint arrays."""
    coords = cp.vertices if pixels is None else pixels
    return [
        np.stack([coords[a], coords[b]])
        for (a, b), kind in zip(cp.edges, cp.assignment)
        if kind != BOUNDARY
    ]


def crease_labels(cp: CreasePattern) -> list[str]:
    return [kind for kind in cp.assignment if kind != BOUNDARY]


def collinear_span(
    reference: np.ndarray,
    candidate: np.ndarray,
    angle_tol: float = np.deg2rad(6.0),
    offset_tol: float = 8.0,
) -> tuple[float, float] | None:
    """The stretch of ``reference`` that a collinear ``candidate`` covers.

    Returned as ``(from, to)`` in distance along ``reference`` from its first
    endpoint, clipped to its extent; ``None`` when the two are not on the same
    line under the given tolerances, and an empty span when they are collinear
    but do not overlap.

    ``match_creases`` needs only the length of this span. Where it sits matters
    to a caller asking why a crease went unmatched: several fragments that each
    fall short of the overlap bar can still cover the crease between them, and
    that is a different failure from nothing being there at all.
    """
    ref_delta = reference[1] - reference[0]
    ref_length = float(np.linalg.norm(ref_delta))
    cand_delta = candidate[1] - candidate[0]
    cand_length = float(np.linalg.norm(cand_delta))
    if ref_length < 1e-9 or cand_length < 1e-9:
        return None

    direction = ref_delta / ref_length
    cosine = abs(float(direction @ (cand_delta / cand_length)))
    if cosine < np.cos(angle_tol):
        return None

    normal = np.array([-direction[1], direction[0]])
    if max(abs((candidate - reference[0]) @ normal)) > offset_tol:
        return None

    t = np.sort((candidate - reference[0]) @ direction)
    start = max(float(t[0]), 0.0)
    return start, max(min(float(t[1]), ref_length), start)


def _overlap(reference: np.ndarray, candidate: np.ndarray, angle_tol: float, offset_tol: float):
    """Overlap length if the two segments lie on the same line, else ``None``."""
    span = collinear_span(reference, candidate, angle_tol, offset_tol)
    return None if span is None else span[1] - span[0]


@dataclass
class MatchResult:
    matched: int
    predicted: int
    truth: int
    mv_correct: int
    # (truth index, predicted index) for every pair that matched. Kept so a
    # caller can ask questions the counts cannot answer -- how far off the
    # recovered line's angle was, for instance.
    pairs: list = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.matched / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.matched / self.truth if self.truth else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def mv_accuracy(self) -> float:
        return self.mv_correct / self.matched if self.matched else float("nan")

    def summary(self) -> str:
        return (
            f"creases P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f} "
            f"({self.matched}/{self.truth} found, {self.predicted} predicted) | "
            f"MV acc={self.mv_accuracy:.3f}"
        )


def match_creases(
    truth: list[np.ndarray],
    truth_labels: list[str],
    predicted: list[np.ndarray],
    predicted_labels: list[str],
    angle_tol: float = np.deg2rad(6.0),
    offset_tol: float = 8.0,
    min_overlap: float = 0.5,
) -> MatchResult:
    """Greedily pair predicted creases with ground-truth ones.

    A ground-truth crease counts as found when some predicted crease lies on the
    same line and covers at least ``min_overlap`` of its length. Each prediction
    is spent on at most one truth, so duplicating a crease costs precision.
    """
    used = set()
    matched = mv_correct = 0
    pairs: list[tuple[int, int]] = []

    for reference_index, (reference, label) in enumerate(zip(truth, truth_labels)):
        length = float(np.linalg.norm(reference[1] - reference[0]))
        if length < 1e-9:
            continue
        best, best_overlap = None, min_overlap * length
        for index, candidate in enumerate(predicted):
            if index in used:
                continue
            overlap = _overlap(reference, candidate, angle_tol, offset_tol)
            if overlap is not None and overlap >= best_overlap:
                best, best_overlap = index, overlap
        if best is not None:
            used.add(best)
            matched += 1
            pairs.append((reference_index, best))
            mv_correct += predicted_labels[best] == label

    return MatchResult(
        matched=matched,
        predicted=len(predicted),
        truth=len(truth),
        mv_correct=mv_correct,
        pairs=pairs,
    )
