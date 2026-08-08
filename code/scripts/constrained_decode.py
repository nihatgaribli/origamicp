#!/usr/bin/env python
"""Does decoding under the fold theorems move the estimate toward the truth?

    python scripts/constrained_decode.py --data ../data/synth \
        --checkpoint ../runs/baseline/best.pt

Everything here is measured against ground truth, never against the constraints
themselves. Enforcing Maekawa and then reporting Maekawa satisfaction is
circular -- it is one by construction and says nothing about whether the answer
got better. The two questions that are not circular:

  MV accuracy      do the corrected labels agree with the true ones more often?
  angular error    do the corrected creases point closer to where they should?

Both are computed only over creases that were matched to a ground-truth crease,
since a spurious line has no true label or angle to be closer to.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from origamicp.decode import constrain_pattern
from origamicp.models import CreaseUNet, SyntheticCreaseDataset, collate, project_to_pixels
from origamicp.vectorize import edge_scores, extract_crease_pattern
from origamicp.vectorize.match import crease_labels, crease_segments, match_creases
from origamicp.verify import verify


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    # Both re-swept on validation by ``scripts/tune.py``; see the notes on the
    # same two flags in ``evaluate_graph.py``.
    parser.add_argument("--threshold", type=float, default=0.92)
    parser.add_argument("--snap", type=float)
    parser.add_argument("--no-angle-projection", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def angular_errors(truth_segments, predicted_segments, pairs) -> list[float]:
    """Angle between each matched pair of lines, in degrees, ignoring sense."""
    errors = []
    for truth_index, predicted_index in pairs:
        a = truth_segments[truth_index][1] - truth_segments[truth_index][0]
        b = predicted_segments[predicted_index][1] - predicted_segments[predicted_index][0]
        norms = np.linalg.norm(a) * np.linalg.norm(b)
        if norms < 1e-9:
            continue
        errors.append(float(np.degrees(np.arccos(np.clip(abs(a @ b) / norms, -1.0, 1.0)))))
    return errors


def evaluate(pattern, truth_segments, truth_labels):
    result = match_creases(
        truth_segments, truth_labels, crease_segments(pattern), crease_labels(pattern)
    )
    return result, angular_errors(truth_segments, crease_segments(pattern), result.pairs)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = CreaseUNet(
        in_channels=3,
        width=checkpoint.get("width", 32),
        depth=checkpoint.get("depth", 4),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = SyntheticCreaseDataset(args.data, args.split, crop=None, jitter=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate)

    raw = {"mv_correct": 0, "matched": 0, "angles": []}
    decoded = {"mv_correct": 0, "matched": 0, "angles": []}
    raw_validity, decoded_validity = [], []
    relabelled = unsatisfiable = conflicts = processed = 0

    from origamicp.core.cp import CreasePattern
    from origamicp.capture import back_face

    with torch.no_grad():
        for index, batch in enumerate(loader):
            row = dataset.rows[index]
            if float(batch["mv_valid"][0]) == 0.0:
                continue

            sheet = batch["sheet"][0].numpy()
            outputs = model(batch["image"].to(device))
            crease_prob = outputs["crease"][0].sigmoid().cpu().numpy() * sheet
            mv_label = outputs["mv"][0].argmax(0).cpu().numpy()

            pattern, _ = extract_crease_pattern(
                crease_prob,
                mv_label,
                sheet,
                threshold=args.threshold,
                **({} if args.snap is None else {"snap": args.snap}),
            )
            if pattern.n_edges == 0:
                continue

            scores = edge_scores(pattern, mv_label, crease_prob)
            corrected, report = constrain_pattern(
                pattern, scores, angle_projection=not args.no_angle_projection
            )

            cp = CreasePattern.from_fold(dataset.root / row["fold_path"])
            if row["face"] == "back":
                cp = back_face(cp)
            corners = np.fromstring(row["corners"], sep=" ").reshape(4, 2)
            truth_segments = crease_segments(cp, project_to_pixels(cp, corners))
            truth_labels = crease_labels(cp)

            for pattern_variant, bucket, validity in (
                (pattern, raw, raw_validity),
                (corrected, decoded, decoded_validity),
            ):
                result, errors = evaluate(pattern_variant, truth_segments, truth_labels)
                bucket["mv_correct"] += result.mv_correct
                bucket["matched"] += result.matched
                bucket["angles"].extend(errors)
                report_v = verify(pattern_variant, tol=np.deg2rad(3.0))
                if report_v.interior:
                    validity.append(report_v.vertex_validity_rate)

            relabelled += report.relabelled
            unsatisfiable += report.unsatisfiable
            conflicts += report.conflicts
            processed += 1
            if args.limit and processed >= args.limit:
                break

    print(f"\n{processed} sheets | {relabelled} creases relabelled | "
          f"{unsatisfiable} unsatisfiable vertices | {conflicts} conflicts\n")
    print(f"{'':<26}{'MV accuracy':>14}{'median angle err':>19}{'mean |angle err|':>19}")
    for name, bucket in (("raw extraction", raw), ("constrained decoding", decoded)):
        accuracy = bucket["mv_correct"] / max(bucket["matched"], 1)
        angles = np.array(bucket["angles"]) if bucket["angles"] else np.array([np.nan])
        print(f"{name:<26}{accuracy:>14.4f}{np.median(angles):>17.3f} deg{np.mean(angles):>17.3f} deg")

    print(
        f"\n(geometric validity {np.mean(raw_validity):.3f} -> {np.mean(decoded_validity):.3f}, "
        "reported only as a check that the constraints were applied -- it is not\n"
        " evidence of improvement, since the decoder enforces exactly what it measures)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
