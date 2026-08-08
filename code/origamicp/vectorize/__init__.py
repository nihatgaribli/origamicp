"""Predicted masks to a crease graph."""

from __future__ import annotations

import numpy as np

from origamicp.core.cp import CreasePattern
from origamicp.vectorize.graph import build_pattern, sheet_polygon
from origamicp.vectorize.scores import edge_scores
from origamicp.vectorize.lines import (
    Segment,
    detect_segments,
    refine_away_from_junctions,
)

REFERENCE_SIZE_PX = 512.0

__all__ = [
    "extract_crease_pattern",
    "detect_segments",
    "refine_away_from_junctions",
    "build_pattern",
    "sheet_polygon",
    "edge_scores",
    "Segment",
]


def extract_crease_pattern(
    crease_prob: np.ndarray,
    mv_label: np.ndarray,
    sheet_mask: np.ndarray,
    threshold: float = 0.5,
    # Swept against the oracle, not guessed -- and swept again at 512 and 768
    # pixels, because whether a parameter should scale with the render is a
    # question about what it measures, not a convention to apply uniformly.
    #
    # ``min_length`` and ``boundary_snap`` are distances on the paper: a crease
    # too short to matter is short in millimetres, so they are quoted for a
    # 512-pixel render and scaled to whatever the image is.
    #
    # ``snap`` and ``min_density`` are not. Snap is how far apart two fitted
    # lines may cross and still be the same junction, which is set by the
    # precision of the fit rather than by anything on the paper; density is
    # pixels of evidence per unit length, and a crease is a band a few pixels
    # wide once there are enough pixels to be a band at all. Scaling snap cost
    # more than it sounds: at 768 it reached 45 px, which merged neighbouring
    # junctions into one vertex and left the oracle recovering 1762 interior
    # vertices where the truth had 1920.
    #
    # Set by ``scripts/tune.py`` on the validation split, selecting on geometric
    # validity. Which metric selects matters more than it looks: validity and
    # crease F1 pull snap in opposite directions, and reading F1 alone picks 25
    # to 30, where validity is 0.46 rather than 0.53 and the extraction returns
    # a fifth more interior vertices than the sheet has. At 35 it returns 1.01x
    # the true count.
    #
    # Fixed, not universal. Snap has a second job -- keeping genuinely separate
    # junctions apart -- and that one is a distance on the paper, so it binds
    # from below as pixels get coarse. ``scripts/resolution.py`` sweeps both per
    # render size and finds the optimum climbing and then flattening (15 at 384,
    # 25 at 768, 30 at 1024); it selects on F1, which is why it lands lower than
    # here. Below roughly 512 px these values are wrong and that sweep is what
    # to trust -- it scores its own configuration rather than importing these.
    min_length: float | None = None,
    # A line threading several junctions collects a vote at each crossing and
    # none in between, so it survives Hough on evidence it never had. That is
    # where the spurious lines came from, and the filter for it was written and
    # then left switched off at zero. Validity is flat across 2.5 to 4.0 and
    # falls away by 4.5, where the bar starts rejecting real crease bands; 3.5
    # is where the model and the oracle agree.
    min_density: float = 3.5,
    snap: float = 30.0,
    boundary_snap: float | None = None,
) -> tuple[CreasePattern, list[Segment]]:
    """Predicted maps to a crease pattern; also returns the fitted segments.

    Returning the segments as well keeps the two failure modes separable when
    something looks wrong: a missing crease is a detection problem, whereas a
    crease that was found but joined to the wrong neighbour is an assembly one.
    """
    scale = sheet_mask.shape[0] / REFERENCE_SIZE_PX
    min_length = 32.0 * scale if min_length is None else min_length
    boundary_snap = 24.0 * scale if boundary_snap is None else boundary_snap

    segments = detect_segments(
        crease_prob,
        sheet_mask,
        threshold=threshold,
        min_length=min_length,
        min_density=min_density,
        band=3.0 * scale,
        max_gap=14.0 * scale,
        rho_tol=6.0 * scale,
        seed_rho_tol=12.0 * scale,
    )
    segments = refine_away_from_junctions(
        segments,
        crease_prob,
        sheet_mask,
        threshold=threshold,
        band=3.0 * scale,
        junction_radius=14.0 * scale,
    )
    polygon = sheet_polygon(sheet_mask)
    pattern = build_pattern(
        segments,
        polygon,
        mv_label,
        crease_prob,
        snap=snap,
        boundary_snap=boundary_snap,
    )
    return pattern, segments
