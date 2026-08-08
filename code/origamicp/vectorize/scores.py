"""Per-crease mountain evidence, read off the model's label map.

``build_pattern`` commits to a hard label per edge. Constrained decoding needs
the strength of that preference too: a vertex where one crease is a coin flip
and three are certain should have the coin flip overruled, not the certainties.
"""

from __future__ import annotations

import numpy as np

from origamicp.core.cp import BOUNDARY, CreasePattern
from origamicp.vectorize.graph import PRED_MOUNTAIN, PRED_VALLEY


def edge_scores(
    cp: CreasePattern,
    mv_label: np.ndarray,
    crease_prob: np.ndarray,
    samples: int = 32,
    trim: float = 0.15,
    offsets: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0),
) -> np.ndarray:
    """Signed evidence per edge: positive for mountain, negative for valley.

    The magnitude is the weighted margin between the two votes, normalised by
    the total weight, so it lies in [-1, 1] and is comparable between a long
    crease and a short one.
    """
    height, width = mv_label.shape
    scores = np.zeros(cp.n_edges, dtype=np.float64)

    for index, ((a, b), kind) in enumerate(zip(cp.edges, cp.assignment)):
        if kind == BOUNDARY:
            continue
        start, end = cp.vertices[a], cp.vertices[b]
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length < 1e-9:
            continue
        normal = np.array([-delta[1], delta[0]]) / length

        t = np.linspace(trim, 1.0 - trim, samples)[:, None]
        points = start * (1 - t) + end * t

        mountain = valley = 0.0
        for offset in offsets:
            shifted = points + normal * offset
            xs = np.clip(np.round(shifted[:, 0]).astype(int), 0, width - 1)
            ys = np.clip(np.round(shifted[:, 1]).astype(int), 0, height - 1)
            labels = mv_label[ys, xs]
            weights = crease_prob[ys, xs].astype(np.float64)
            mountain += float(weights[labels == PRED_MOUNTAIN].sum())
            valley += float(weights[labels == PRED_VALLEY].sum())

        total = mountain + valley
        scores[index] = (mountain - valley) / total if total > 1e-9 else 0.0
    return scores
