"""Front/back face transforms.

Scanning both faces of a sheet doubles the dataset for free: the back scan is a
mirrored view of the same paper, and every mountain seen from the front is a
valley seen from the back. That relabelling is exact, not an approximation, so
back-face ground truth costs zero extra annotation.

It also buys us a diagnostic no existing origami benchmark has. A model that
reads the physics must invert its MV prediction when shown the other face; a
model that has latched onto texture or ink statistics will not. See
``mv_flip_consistency``.
"""

from __future__ import annotations

import numpy as np

from origamicp.core.cp import MOUNTAIN, VALLEY, CreasePattern

_MV_SWAP = {MOUNTAIN: VALLEY, VALLEY: MOUNTAIN}


def flip_assignment(assignment: np.ndarray) -> np.ndarray:
    """Swap M and V, leaving boundary/flat/unassigned edges untouched."""
    assignment = np.asarray(assignment, dtype="<U1")
    return np.array([_MV_SWAP.get(a, a) for a in assignment], dtype="<U1")


def mirror_vertices(vertices: np.ndarray, axis: int = 0) -> np.ndarray:
    """Reflect coordinates about the mid-line of their own bounding box.

    Reflecting about the bounding box (rather than about x=0) keeps the sheet
    in place, which is what we want when the back scan is framed the same way
    as the front one.
    """
    vertices = np.array(vertices, dtype=np.float64, copy=True)
    lo, hi = vertices[:, axis].min(), vertices[:, axis].max()
    vertices[:, axis] = lo + hi - vertices[:, axis]
    return vertices


def back_face(cp: CreasePattern, axis: int = 0) -> CreasePattern:
    """Ground truth for the reverse side of the same physical sheet."""
    return CreasePattern(
        mirror_vertices(cp.vertices, axis), cp.edges, flip_assignment(cp.assignment)
    )


def mv_flip_consistency(front_pred: np.ndarray, back_pred: np.ndarray) -> float:
    """Fraction of creases whose predicted MV correctly inverts across faces.

    Both arrays must list the same creases in the same order. Only edges the
    model committed to as M or V on *both* faces are scored; a model that hedges
    with U everywhere neither gains nor loses here, so report coverage too.

    A model can score well on per-edge MV accuracy while scoring near chance
    here, which is exactly the failure mode worth reporting.
    """
    front = np.asarray(front_pred, dtype="<U1")
    back = np.asarray(back_pred, dtype="<U1")
    if front.shape != back.shape:
        raise ValueError(f"shape mismatch: {front.shape} vs {back.shape}")

    scorable = np.isin(front, [MOUNTAIN, VALLEY]) & np.isin(back, [MOUNTAIN, VALLEY])
    if not scorable.any():
        return float("nan")
    return float(np.mean(front[scorable] != back[scorable]))
