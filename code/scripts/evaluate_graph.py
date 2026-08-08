#!/usr/bin/env python
"""Evaluate the full pipeline: image -> mask -> crease graph -> geometry check.

    python scripts/evaluate_graph.py --data ../data/synth \
        --checkpoint ../runs/baseline/best.pt

Reports three levels, because they answer different questions and can disagree:

  pixel     how well the mask lines up, sensitive to a one-pixel shift
  crease    whether each fold was recovered as a line, which is what folding needs
  geometry  whether the recovered graph satisfies Maekawa, Kawasaki and
            big-little-big -- the metric this project exists to report

The geometry number is expected to be the harshest. A single missed crease
leaves a vertex with the wrong degree and fails it outright, so it measures
whether the whole pattern came back rather than most of it.

An oracle run (--oracle) puts ground-truth masks through the same vectoriser.
That separates the model's errors from the vectoriser's: whatever the oracle
fails to recover, no amount of training will fix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from origamicp.core.cp import CreasePattern
from origamicp.models import (
    Counts,
    CreaseUNet,
    SyntheticCreaseDataset,
    build_targets,
    collate,
    project_to_pixels,
)
from origamicp.vectorize import extract_crease_pattern
from origamicp.vectorize.match import MatchResult, crease_labels, crease_segments, match_creases
from origamicp.verify import verify


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    # Re-swept on validation by ``scripts/tune.py`` once the density filter was
    # switched on, since the two interact: the earlier 0.85 was chosen when
    # nothing downstream removed a spurious line, so the threshold had to keep
    # the mask clean by itself. The filter doing that job does not buy a lower
    # threshold -- 0.50 and 0.65 are the worst settings in the sweep. A thicker
    # mask fuses neighbouring creases into one blob, and structure loses more
    # there than the filter recovers. Note that 0.75, 0.85 and 0.92 are not
    # cleanly separated at 60 sheets; 0.92 is the argmax on validity, not a
    # result that would survive a significance test.
    parser.add_argument("--threshold", type=float, default=0.92)
    # Left as None so the tuned library defaults apply. Repeating the numbers
    # here once let them drift apart, and the script quietly evaluated a
    # configuration nobody had swept.
    parser.add_argument("--min-length", type=float)
    parser.add_argument("--min-density", type=float)
    parser.add_argument("--snap", type=float)
    parser.add_argument("--boundary-snap", type=float)
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="vectorise ground-truth masks instead of predictions",
    )
    parser.add_argument(
        "--classical",
        action="store_true",
        help="learning-free directional-step baseline instead of the network",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def truth_pattern(dataset: SyntheticCreaseDataset, row: dict) -> tuple[CreasePattern, np.ndarray]:
    cp = CreasePattern.from_fold(dataset.root / row["fold_path"])
    if row["face"] == "back":
        from origamicp.capture import back_face

        cp = back_face(cp)
    corners = np.fromstring(row["corners"], sep=" ").reshape(4, 2)
    return cp, corners


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.oracle and not args.classical and args.checkpoint is None:
        print("need --checkpoint unless --oracle is given")
        return 1

    device = torch.device(args.device)
    dataset = SyntheticCreaseDataset(args.data, args.split, crop=None, jitter=False)

    model = None
    if not args.oracle and not args.classical:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = CreaseUNet(
            in_channels=3,
            width=checkpoint.get("width", 32),
            depth=checkpoint.get("depth", 4),
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()

    pixel = Counts()
    totals = MatchResult(0, 0, 0, 0)
    validity, kawasaki, maekawa, blb = [], [], [], []
    interior_truth, interior_found = 0, 0
    processed = 0

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate)
    for index, batch in enumerate(loader):
        row = dataset.rows[index]
        if float(batch["mv_valid"][0]) == 0.0:
            continue  # back-lit sheets carry no MV to score

        sheet = batch["sheet"][0].numpy()
        if args.classical:
            from origamicp.models.classical import classical_predict

            gray = (batch["image"][0, 0].numpy() * 255).astype(np.uint8)
            crease_prob, mv_label = classical_predict(
                gray, sheet, float(row["light_azimuth_deg"]), threshold=args.threshold
            )
        elif args.oracle:
            crease_prob = batch["crease"][0].numpy().astype(np.float32)
            mv_label = np.where(batch["mv"][0].numpy() == 1, 0, 1).astype(np.int64)
        else:
            with torch.no_grad():
                tensors = {
                    k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
                }
                outputs = model(tensors["image"])
                pixel.update(outputs, tensors, args.threshold)
            crease_prob = outputs["crease"][0].sigmoid().cpu().numpy() * sheet
            mv_label = outputs["mv"][0].argmax(0).cpu().numpy()

        overrides = {
            name: value
            for name, value in (
                ("min_length", args.min_length),
                ("min_density", args.min_density),
                ("snap", args.snap),
                ("boundary_snap", args.boundary_snap),
            )
            if value is not None
        }
        predicted, _ = extract_crease_pattern(
            crease_prob, mv_label, sheet, threshold=args.threshold, **overrides
        )

        cp, corners = truth_pattern(dataset, row)
        truth_pixels = project_to_pixels(cp, corners)
        result = match_creases(
            crease_segments(cp, truth_pixels),
            crease_labels(cp),
            crease_segments(predicted),
            crease_labels(predicted),
        )
        totals = MatchResult(
            totals.matched + result.matched,
            totals.predicted + result.predicted,
            totals.truth + result.truth,
            totals.mv_correct + result.mv_correct,
        )

        report = verify(predicted, tol=np.deg2rad(3.0))
        if report.interior:
            validity.append(report.vertex_validity_rate)
            kawasaki.append(report.kawasaki_rate)
            maekawa.append(report.maekawa_rate)
            blb.append(report.blb_rate)
        interior_truth += len(cp.interior_vertices())
        interior_found += len(report.interior)

        processed += 1
        if args.limit and processed >= args.limit:
            break

    label = (
        "oracle (ground-truth masks)"
        if args.oracle
        else "classical directional-step baseline"
        if args.classical
        else f"model {args.checkpoint.name}"
    )
    print(f"\n{label} on {args.split}: {processed} sheets\n")
    if not args.oracle and not args.classical:
        print(f"  pixel     {pixel.summary()}")
    print(f"  crease    {totals.summary()}")
    print(
        f"  geometry  valid vertices {np.mean(validity):.3f} | "
        f"kawasaki {np.mean(kawasaki):.3f} | maekawa {np.mean(maekawa):.3f} | "
        f"big-little-big {np.mean(blb):.3f}"
        if validity
        else "  geometry  no interior vertices recovered"
    )
    print(f"  vertices  {interior_found} interior recovered vs {interior_truth} in truth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
