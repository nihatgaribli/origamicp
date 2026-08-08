#!/usr/bin/env python
"""Choose the pipeline's free parameters on validation, and record the sweep.

    python scripts/tune.py --data ../data/synth768 --oracle
    python scripts/tune.py --data ../data/synth768 --checkpoint ../runs/b768/best.pt

Every parameter below was at some point tuned against a pipeline that no longer
exists. ``min_density`` was swept, documented, and then left at zero;
``snap`` was swept at 512 and afterwards scaled by render size, which made it
wrong everywhere else; ``threshold`` was swept while ``min_density`` was zero,
so it was chosen to keep masks clean by itself at a time when nothing
downstream would remove a spurious line. Each was defensible when set and stale
by the time it was used.

They are swept here together, on the validation split, so the choice is on
record and can be repeated when the pipeline changes again. Nothing in this
script touches the test split -- an earlier round of this work chose defaults by
watching test numbers, which is exactly what this exists to stop.

Two scores are reported because they disagree, and the disagreement is the
finding the project is built around: crease F1 asks whether the folds were
found, geometric validity asks whether what came back could be folded. The
selection rule is geometric validity, since that is the metric the write-up
leads with, and the table prints both so the cases where F1 would have chosen
differently stay visible.

The model runs once per image and every configuration is scored on the same
cached probability map, so the grid costs extraction time rather than inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from origamicp.models import CreaseUNet, SyntheticCreaseDataset, collate, project_to_pixels
from origamicp.models.classical import classical_predict
from origamicp.vectorize import extract_crease_pattern
from origamicp.vectorize.match import crease_labels, crease_segments, match_creases
from origamicp.verify import verify
from scripts.evaluate_graph import truth_pattern

# Thresholding a ground-truth mask is a no-op, so the oracle sweeps only the two
# vectoriser parameters and the model adds the one that turns its output into a
# mask in the first place.
GRID_THRESHOLD = (0.50, 0.65, 0.75, 0.85, 0.92)
# The baseline thresholds its own directional-step response, which is scaled
# differently from a sigmoid, so its useful range starts lower.
GRID_THRESHOLD_CLASSICAL = (0.25, 0.35, 0.50, 0.60, 0.70, 0.80)
# Reaching past where the oracle turns over (40), because the model need not
# share its optimum: a noisy mask scatters junctions that a wider merge radius
# pulls back together, and the pre-fix pipeline reached higher model validity at
# an effective 45 than anything measured since.
GRID_SNAP = (20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0)
# Up to the point where the filter starts taking real creases: past about 4.5
# the density of a genuine crease band no longer clears the bar and the
# extraction empties out, so the grid covers the approach to that edge.
GRID_DENSITY = (0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--oracle", action="store_true", help="sweep against ground-truth masks"
    )
    # The baseline is entitled to the same search the model got. Its masks are
    # noisier, so a density filter set against clean ones cuts into real creases
    # -- and a comparison quoted from that is a comparison with a handicap the
    # other side chose.
    parser.add_argument(
        "--classical", action="store_true", help="sweep the learning-free baseline"
    )
    parser.add_argument(
        "--snap",
        type=float,
        help="hold snap fixed instead of sweeping it (for the model run, once "
        "the oracle has settled it)",
    )
    # Narrowing the grid is what makes the full split affordable. A sweep over
    # sixty sheets chose values that reversed on test, so the split matters more
    # than the breadth of the grid: better to settle one axis at a time over
    # everything than three at once over a quarter of it.
    parser.add_argument(
        "--density", type=float, help="hold min_density fixed instead of sweeping it"
    )
    parser.add_argument(
        "--threshold", type=float, help="hold the mask threshold fixed instead of sweeping it"
    )
    # The last parameter never swept. It still scales with the render, on the
    # same reasoning that turned out to be wrong for snap, so it is worth one
    # run per value rather than an assumption.
    parser.add_argument(
        "--boundary-snap", type=float, help="override the boundary snapping radius"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


class _Totals:
    def __init__(self) -> None:
        self.matched = self.predicted = self.truth = 0
        self.validity: list[float] = []
        self.interior = self.interior_truth = 0

    def add(self, result, report, interior_truth: int) -> None:
        self.matched += result.matched
        self.predicted += result.predicted
        self.truth += result.truth
        if report.interior:
            self.validity.append(report.vertex_validity_rate)
        self.interior += len(report.interior)
        # Validity is an average over the vertices that came back, so it can be
        # bought by returning fewer of them. Carrying the true count alongside
        # is what separates a configuration that recovers the pattern from one
        # that merges it down to the handful of vertices it can satisfy.
        self.interior_truth += interior_truth

    @property
    def f1(self) -> float:
        precision = self.matched / max(self.predicted, 1)
        recall = self.matched / max(self.truth, 1)
        return 2 * precision * recall / max(precision + recall, 1e-9)

    @property
    def mean_validity(self) -> float:
        return float(np.mean(self.validity)) if self.validity else 0.0


def main(argv=None) -> int:
    args = parse_args(argv)
    if not (args.oracle or args.classical) and args.checkpoint is None:
        print("need --checkpoint unless --oracle or --classical is given")
        return 1
    if args.split == "test":
        print("refusing to tune on test")
        return 1

    device = torch.device(args.device)
    dataset = SyntheticCreaseDataset(args.data, args.split, crop=None, jitter=False)

    model = None
    if not (args.oracle or args.classical):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = CreaseUNet(
            in_channels=3,
            width=checkpoint.get("width", 32),
            depth=checkpoint.get("depth", 4),
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()

    if args.oracle:
        thresholds = (0.5,)  # thresholding a ground-truth mask is a no-op
    elif args.threshold is not None:
        thresholds = (args.threshold,)
    elif args.classical:
        thresholds = GRID_THRESHOLD_CLASSICAL
    else:
        thresholds = GRID_THRESHOLD
    snaps = (args.snap,) if args.snap is not None else GRID_SNAP
    densities = (args.density,) if args.density is not None else GRID_DENSITY
    combos = [(t, s, d) for t in thresholds for s in snaps for d in densities]
    totals = {combo: _Totals() for combo in combos}

    processed = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate)
    for index, batch in enumerate(loader):
        row = dataset.rows[index]
        if float(batch["mv_valid"][0]) == 0.0:
            continue

        sheet = batch["sheet"][0].numpy()
        # The baseline recomputes its map for every threshold, so it is cached
        # per value rather than per combination -- the density axis reuses it.
        classical_cache: dict[float, tuple] = {}
        if args.classical:
            crease_prob = mv_label = None
        elif model is None:
            crease_prob = batch["crease"][0].numpy().astype(np.float32)
            mv_label = np.where(batch["mv"][0].numpy() == 1, 0, 1).astype(np.int64)
        else:
            with torch.no_grad():
                outputs = model(batch["image"].to(device))
            crease_prob = outputs["crease"][0].sigmoid().cpu().numpy() * sheet
            mv_label = outputs["mv"][0].argmax(0).cpu().numpy()

        cp, corners = truth_pattern(dataset, row)
        truth_segments = crease_segments(cp, project_to_pixels(cp, corners))
        truth_labels = crease_labels(cp)

        for combo in combos:
            threshold, snap, density = combo
            if args.classical:
                if threshold not in classical_cache:
                    gray = (batch["image"][0, 0].numpy() * 255).astype(np.uint8)
                    classical_cache[threshold] = classical_predict(
                        gray, sheet, float(row["light_azimuth_deg"]), threshold=threshold
                    )
                crease_prob, mv_label = classical_cache[threshold]
            pattern, _ = extract_crease_pattern(
                crease_prob,
                mv_label,
                sheet,
                threshold=threshold,
                snap=snap,
                min_density=density,
                **(
                    {}
                    if args.boundary_snap is None
                    else {"boundary_snap": args.boundary_snap}
                ),
            )
            totals[combo].add(
                match_creases(
                    truth_segments,
                    truth_labels,
                    crease_segments(pattern),
                    crease_labels(pattern),
                ),
                verify(pattern, tol=np.deg2rad(3.0)),
                len(cp.interior_vertices()),
            )

        processed += 1
        if args.limit and processed >= args.limit:
            break

    source = (
        "ground-truth masks"
        if args.oracle
        else "learning-free baseline"
        if args.classical
        else f"model {args.checkpoint.name}"
    )
    print(f"\nsweep on {source}, {args.split} split: {processed} sheets\n")
    header = (
        f"{'threshold':>10}{'snap':>7}{'density':>9}{'crease F1':>11}"
        f"{'validity':>10}{'vertices':>10}{'vs truth':>10}"
    )
    print(header)

    ranked = sorted(totals.items(), key=lambda item: -item[1].mean_validity)
    for (threshold, snap, density), total in ranked:
        ratio = total.interior / max(total.interior_truth, 1)
        print(
            f"{threshold:>10.2f}{snap:>7.0f}{density:>9.1f}"
            f"{total.f1:>11.3f}{total.mean_validity:>10.3f}{total.interior:>10}"
            f"{ratio:>9.2f}x"
        )

    best_validity = ranked[0]
    best_f1 = max(totals.items(), key=lambda item: item[1].f1)
    print(
        f"\n  best validity  threshold={best_validity[0][0]:.2f} "
        f"snap={best_validity[0][1]:.0f} density={best_validity[0][2]:.1f}"
        f"  -> validity {best_validity[1].mean_validity:.3f}, F1 {best_validity[1].f1:.3f}"
    )
    print(
        f"  best crease F1 threshold={best_f1[0][0]:.2f} "
        f"snap={best_f1[0][1]:.0f} density={best_f1[0][2]:.1f}"
        f"  -> validity {best_f1[1].mean_validity:.3f}, F1 {best_f1[1].f1:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
