"""A learning-free baseline built directly from the shading physics.

Worth having for two reasons. It sets the bar the network has to clear -- if a
directional derivative matches it, the network is not earning its place. And it
makes the mountain/valley rule explicit: given where the light is, the sign of
the brightness step across a crease *is* the answer, with no training involved.

That second point is the same fact the light ablation demonstrates from the
other side. Here the light direction is used analytically; there it is withheld
and the learned model collapses to chance. Both say the assignment lives in the
relationship between the crease and the lamp, not in the crease alone.

One operator does both jobs, which is not a coincidence. Under raking light a
crease is not a symmetric ridge in the image -- it is bright on one side and
dark on the other, an odd feature. So the derivative along the light direction
peaks at the crease centre (detection) and carries the fold direction in its
sign (labelling). An unsigned ridge filter, by contrast, peaks on the flanks
and throws the sign away.
"""

from __future__ import annotations

import cv2
import numpy as np

from origamicp.vectorize.graph import PRED_MOUNTAIN, PRED_VALLEY


def directional_step(
    gray: np.ndarray, light_azimuth_deg: float, offset_px: float = 3.5, smooth: float = 1.5
) -> np.ndarray:
    """Brightness toward the lamp minus brightness away from it, per pixel."""
    smoothed = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), smooth)
    angle = np.deg2rad(light_azimuth_deg)
    towards = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

    height, width = gray.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    sampled = []
    for sign in (1.0, -1.0):
        sampled.append(
            cv2.remap(
                smoothed,
                grid_x + sign * towards[0] * offset_px,
                grid_y + sign * towards[1] * offset_px,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        )
    return sampled[0] - sampled[1]


def classical_predict(
    image: np.ndarray,
    sheet_mask: np.ndarray,
    light_azimuth_deg: float | None,
    threshold: float = 0.35,
    offset_px: float = 3.5,
    smooth: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect and label creases from the directional brightness step.

    Returns ``(crease_prob, mv_label)`` in the network's output format -- the
    two-way ``PRED_*`` convention, not the three-way target map -- so the same
    evaluation code scores both.

    ``light_azimuth_deg`` of ``None`` means the direction is unknown, as in
    back-lit capture. Everything is then called a mountain: without the lamp the
    sign is not recoverable, and guessing per crease would manufacture a coin
    flip that looks like a decision.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalise inside the sheet only. The paper-to-backing edge is a far larger
    # step than any crease, so a percentile taken over the whole frame is set by
    # the border and scales every crease down to nothing.
    inside = cv2.erode((sheet_mask > 0).astype(np.uint8), np.ones((9, 9), np.uint8))

    if light_azimuth_deg is None:
        from origamicp.capture.register import ridge_response

        magnitude = ridge_response(gray)
        mv_label = np.full(gray.shape, PRED_MOUNTAIN, dtype=np.int64)
    else:
        step = directional_step(gray, light_azimuth_deg, offset_px, smooth)
        magnitude = np.abs(step)
        mv_label = np.where(step > 0, PRED_MOUNTAIN, PRED_VALLEY).astype(np.int64)

    interior = magnitude[inside > 0]
    scale = float(np.percentile(interior, 99.0)) if interior.size else 0.0
    response = np.clip(magnitude / max(scale, 1e-6), 0.0, 1.0)

    crease_prob = (response * (inside > 0)).astype(np.float32)
    crease_prob[crease_prob < threshold] = 0.0
    return crease_prob, mv_label
