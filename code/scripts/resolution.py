#!/usr/bin/env python
"""How much capture resolution the task actually needs.

    python scripts/resolution.py --out resolution.png

Runs the oracle -- ground-truth masks through the real vectoriser -- at a range
of render sizes. Using the oracle isolates the question: no network is involved,
so whatever changes is the resolution alone.

That last sentence is only true if the vectoriser is equally well tuned at every
size, and an earlier version of this study did not check. It ran one fixed
configuration across the range, and two of those parameters were wrong away from
the size they had been swept at. The curve it produced flattened above 112 dpi,
which read as the task saturating; with the parameters corrected it keeps
climbing to the end of the range. The flattening was the vectoriser going out of
tune, not the paper running out of detail.

So the sweep is part of the measurement now. At each size the vectoriser's two
resolution-sensitive parameters are swept, and the best configuration is the one
reported. Otherwise the curve mixes two effects that point in opposite
directions -- coarser pixels lose creases, and a mistuned vectoriser loses them
faster -- and there is no way to read which is which.

Tuning and reporting use different designs. Choosing a configuration on the same
sheets it is scored on buys a few points of F1 that would not survive contact
with new data, and it buys more of them where the optimum is sharp, which bends
the curve rather than merely lifting it.

Both curves are reported: tuned per size, and the library defaults everywhere.
The gap between them is how much of the shape belongs to the vectoriser rather
than to the capture, which is the thing the earlier version could not show.

Only ``snap`` and ``min_density`` are swept -- the two known to be pixel
quantities rather than distances on the paper. ``min_length`` and
``boundary_snap`` scale with the render and are left alone; whether that is
right for ``boundary_snap`` has not been tested.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from origamicp.generate.designs import SHEET_MM
from origamicp.generate.random_cp import random_design
from origamicp.models import build_targets, project_to_pixels
from origamicp.render import ScanStyle, render
from origamicp.vectorize import extract_crease_pattern
from origamicp.vectorize.match import crease_labels, crease_segments, match_creases

REFERENCE_SIZE_PX = 512.0

# Swept jointly: they trade against each other. A stricter density filter leaves
# fewer lines to cross, which changes how far apart two crossings have to be
# before they are separate junctions.
GRID_SNAP = (15.0, 20.0, 25.0, 30.0, 40.0)
GRID_DENSITY = (1.0, 2.0, 2.5, 3.0, 3.5, 4.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 384, 512, 768, 1024])
    parser.add_argument("--n", type=int, default=40, help="designs scored per size")
    # A thirty-point grid picks up a lot of noise from a handful of designs. At
    # twelve the sweep chose a configuration that lost to the library default on
    # held-out sheets, which is the sweep overfitting rather than a real
    # optimum; the reported default curve is what catches that.
    parser.add_argument("--n-tune", type=int, default=24, help="designs the sweep sees")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=Path, default=Path("resolution.png"))
    return parser.parse_args(argv)


def _sample(rng: np.random.Generator, size: int, count: int) -> list[tuple]:
    """Render ``count`` designs once, so every configuration scores the same paper."""
    samples = []
    for _ in range(count):
        _, cp = random_design(rng)
        _, corners = render(cp, ScanStyle(size_px=size), np.random.default_rng(7))
        targets = build_targets(cp, corners, size)
        samples.append(
            (
                cp,
                corners,
                targets["crease"].astype(np.float32),
                np.where(targets["mv"] == 1, 0, 1).astype(np.int64),
                targets["sheet"],
            )
        )
    return samples


def _score(samples: list[tuple], size: int, **overrides) -> tuple[float, float, float]:
    matched = predicted = truth = 0
    for cp, corners, crease_prob, mv_label, sheet in samples:
        pattern, _ = extract_crease_pattern(crease_prob, mv_label, sheet, **overrides)
        result = match_creases(
            crease_segments(cp, project_to_pixels(cp, corners)),
            crease_labels(cp),
            crease_segments(pattern),
            crease_labels(pattern),
            # The matching tolerance is a physical distance, so it has to
            # scale too, or the comparison rewards coarse renders.
            offset_tol=8.0 * size / REFERENCE_SIZE_PX,
        )
        matched += result.matched
        predicted += result.predicted
        truth += result.truth

    recall = matched / max(truth, 1)
    precision = matched / max(predicted, 1)
    return recall, precision, 2 * precision * recall / max(precision + recall, 1e-9)


def main(argv=None) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args(argv)
    rows = []

    header = (
        f"{'render px':>10}{'px/mm':>8}{'dpi':>7}"
        f"{'recall':>9}{'precision':>11}{'F1':>8}"
        f"{'snap':>7}{'density':>9}{'F1 default':>12}"
    )
    print(header)
    for size in args.sizes:
        rng = np.random.default_rng(args.seed)
        # The tuning designs are drawn first and the scored ones after, from the
        # same stream, so the two sets never share a design.
        tuning = _sample(rng, size, args.n_tune)
        scored = _sample(rng, size, args.n)

        best = None
        for snap in GRID_SNAP:
            for density in GRID_DENSITY:
                _, _, f1 = _score(tuning, size, snap=snap, min_density=density)
                if best is None or f1 > best[0]:
                    best = (f1, snap, density)
        _, snap, density = best

        recall, precision, f1 = _score(scored, size, snap=snap, min_density=density)
        _, _, f1_default = _score(scored, size)

        per_mm = size * (1 - 2 * ScanStyle().margin_frac) / SHEET_MM
        rows.append((size, per_mm, recall, precision, f1, snap, density, f1_default))
        print(
            f"{size:>10}{per_mm:>8.1f}{per_mm * 25.4:>7.0f}"
            f"{recall:>9.3f}{precision:>11.3f}{f1:>8.3f}"
            f"{snap:>7.0f}{density:>9.1f}{f1_default:>12.3f}",
            flush=True,
        )

    sizes, per_mm, recalls, precisions, f1s, snaps, densities, defaults = map(
        np.array, zip(*rows)
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(per_mm, recalls, "o-", color="#3b6ea5", lw=2, label="crease recall (tuned)")
    ax.plot(per_mm, precisions, "s-", color="#c1272d", lw=2, label="crease precision (tuned)")
    ax.plot(per_mm, f1s, "^-", color="#2e7d32", lw=2, label="crease F1 (tuned)")
    ax.plot(
        per_mm,
        defaults,
        "^--",
        color="#2e7d32",
        lw=1.6,
        alpha=0.55,
        label="crease F1 (one fixed configuration)",
    )
    ax.set_xlabel("capture resolution (pixels per mm of paper)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    ax.set_title(
        "Ground-truth masks through the vectoriser: what resolution alone costs",
        fontsize=12,
    )
    secondary = ax.secondary_xaxis("top", functions=(lambda v: v * 25.4, lambda v: v / 25.4))
    secondary.set_xlabel("dpi")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nfigure -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
