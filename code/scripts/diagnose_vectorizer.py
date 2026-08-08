#!/usr/bin/env python
"""Ask *why* each crease was missed, rather than how many were.

    python scripts/diagnose_vectorizer.py --data ../data/synth768

From perfect masks the vectoriser reaches crease F1 0.591: 6317 lines predicted
against 5444 true creases, of which only 3474 match. Precision and recall fail
together, which a pure detection shortfall does not do -- missing a crease costs
recall and leaves precision alone. Something is turning one crease into several
predictions that each fall short of the overlap bar, and that pays twice.

Two hypotheses have already been tested against this shortfall and had no
effect: the minimum-length threshold discarding short creases, and the seed
clustering merging near-parallel lines. Guessing a third time is the wrong move,
so this script classifies every unmatched crease instead of proposing a cause.

The split that matters is between the two stages, because they fail for
unrelated reasons and the fix lives in one of them:

  detection   Hough seeds refit to straight lines (``detect_segments``)
  assembly    those lines split at their crossings into a graph (``build_pattern``)

``extract_crease_pattern`` returns both, so each unmatched crease can be asked
whether the line fitter ever found it, and if it did, what the graph did to it.
Categories, in the order they are tested:

  contested    a graph edge covers it, but greedy matching spent that edge on a
               different crease -- two truths competing for one prediction
  fragmented   graph edges cover it between them, none alone past the bar --
               split at crossings the truth does not have
  dropped      the fitter had it, the graph does not -- pruned or merged away
  partial      the fitter found only part of it -- extent or offset wrong
  undetected   nothing collinear anywhere -- a real detection failure

Predictions that match nothing are classified the same way from the other side,
which is where the 2843 unmatched predictions are accounted for.

Runs on ground-truth masks by default: no checkpoint, no training, so a change
to the vectoriser can be measured in minutes.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# Reaching the sibling script by name, as the tests do, rather than copying its
# seven-line truth loader: the back-face flip it applies is easy to forget, and
# a diagnosis that read the truth differently from the evaluation would be worse
# than no diagnosis.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from origamicp.models import CreaseUNet, SyntheticCreaseDataset, collate, project_to_pixels
from origamicp.vectorize import extract_crease_pattern
from origamicp.vectorize.match import (
    collinear_span,
    crease_labels,
    crease_segments,
    match_creases,
)
from scripts.evaluate_graph import truth_pattern

# The bar match_creases sets: a prediction must cover half the true crease.
MIN_OVERLAP = 0.5
# Below this, "the fitter found part of it" becomes "the fitter found nothing" --
# a sliver of collinear pixels is not evidence the crease was detected.
TRACE = 0.1

TRUTH_CATEGORIES = ["contested", "fragmented", "dropped", "partial", "undetected"]
PREDICTION_CATEGORIES = ["fragment", "duplicate", "spurious"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="diagnose model predictions instead of ground-truth masks",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def _covered(reference: np.ndarray, candidates: list[np.ndarray]) -> tuple[float, float, int]:
    """Longest single cover, union of all covers, and how many contribute.

    The union is what separates a crease chopped into pieces from one that was
    never found: the pieces still cover it, they just do so severally.
    """
    spans = []
    for candidate in candidates:
        span = collinear_span(reference, candidate)
        if span is not None and span[1] - span[0] > 1e-9:
            spans.append(span)
    if not spans:
        return 0.0, 0.0, 0

    longest = max(hi - lo for lo, hi in spans)
    union, reach = 0.0, -np.inf
    for lo, hi in sorted(spans):
        union += hi - max(lo, reach)
        reach = max(reach, hi)
    return longest, union, len(spans)


def classify_truth(
    reference: np.ndarray,
    graph_segments: list[np.ndarray],
    raw_segments: list[np.ndarray],
) -> tuple[str, int]:
    """Why this true crease went unmatched, and how many pieces cover it."""
    length = float(np.linalg.norm(reference[1] - reference[0]))
    bar = MIN_OVERLAP * length

    graph_longest, graph_union, pieces = _covered(reference, graph_segments)
    if graph_longest >= bar:
        # A prediction does cover it, so the only way it went unmatched is that
        # the greedy pass spent that prediction on a different crease.
        return "contested", pieces
    if graph_union >= bar:
        return "fragmented", pieces

    _, raw_union, _ = _covered(reference, raw_segments)
    if raw_union >= bar:
        return "dropped", pieces
    if raw_union >= TRACE * length:
        return "partial", pieces
    return "undetected", pieces


def classify_prediction(
    candidate: np.ndarray, truth_segments: list[np.ndarray], matched_truth: set[int]
) -> str:
    """Why this prediction matched nothing."""
    best_fraction, best_index = 0.0, None
    for index, reference in enumerate(truth_segments):
        span = collinear_span(reference, candidate)
        if span is None:
            continue
        length = float(np.linalg.norm(reference[1] - reference[0]))
        if length < 1e-9:
            continue
        fraction = (span[1] - span[0]) / length
        if fraction > best_fraction:
            best_fraction, best_index = fraction, index

    if best_index is None or best_fraction <= 0.0:
        return "spurious"
    if best_fraction >= MIN_OVERLAP and best_index in matched_truth:
        # Covers a crease that some other prediction was matched to first.
        return "duplicate"
    return "fragment"


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    dataset = SyntheticCreaseDataset(args.data, args.split, crop=None, jitter=False)

    model = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = CreaseUNet(
            in_channels=3,
            width=checkpoint.get("width", 32),
            depth=checkpoint.get("depth", 4),
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()

    truth_counts: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    fragment_pieces: list[int] = []
    total_truth = total_predicted = total_matched = 0
    processed = 0

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate)
    for index, batch in enumerate(loader):
        row = dataset.rows[index]
        if float(batch["mv_valid"][0]) == 0.0:
            continue

        sheet = batch["sheet"][0].numpy()
        if model is None:
            crease_prob = batch["crease"][0].numpy().astype(np.float32)
            mv_label = np.where(batch["mv"][0].numpy() == 1, 0, 1).astype(np.int64)
        else:
            with torch.no_grad():
                outputs = model(batch["image"].to(device))
            crease_prob = outputs["crease"][0].sigmoid().cpu().numpy() * sheet
            mv_label = outputs["mv"][0].argmax(0).cpu().numpy()

        predicted, segments = extract_crease_pattern(
            crease_prob, mv_label, sheet, threshold=args.threshold
        )
        raw_segments = [np.stack([s.start, s.end]) for s in segments]

        cp, corners = truth_pattern(dataset, row)
        truth_segments = crease_segments(cp, project_to_pixels(cp, corners))
        graph_segments = crease_segments(predicted)

        result = match_creases(
            truth_segments, crease_labels(cp), graph_segments, crease_labels(predicted)
        )
        matched_truth = {t for t, _ in result.pairs}
        matched_predictions = {p for _, p in result.pairs}

        for truth_index, reference in enumerate(truth_segments):
            if truth_index in matched_truth:
                continue
            category, pieces = classify_truth(reference, graph_segments, raw_segments)
            truth_counts[category] += 1
            if category == "fragmented":
                fragment_pieces.append(pieces)

        for prediction_index, candidate in enumerate(graph_segments):
            if prediction_index in matched_predictions:
                continue
            prediction_counts[classify_prediction(candidate, truth_segments, matched_truth)] += 1

        total_truth += result.truth
        total_predicted += result.predicted
        total_matched += result.matched

        processed += 1
        if args.limit and processed >= args.limit:
            break

    source = "ground-truth masks" if model is None else f"model {args.checkpoint.name}"
    missed = total_truth - total_matched
    unmatched = total_predicted - total_matched

    print(f"\nvectoriser diagnosis on {source}: {processed} sheets\n")
    print(
        f"  {total_matched} of {total_truth} creases matched, "
        f"{total_predicted} predicted "
        f"(P={total_matched / max(total_predicted, 1):.3f} "
        f"R={total_matched / max(total_truth, 1):.3f})\n"
    )

    print(f"  why {missed} true creases went unmatched")
    for category in TRUTH_CATEGORIES:
        count = truth_counts[category]
        print(f"    {category:<12} {count:>6}   {count / max(missed, 1):>6.1%}")

    if fragment_pieces:
        print(
            f"\n    fragmented creases are split into "
            f"{np.mean(fragment_pieces):.1f} pieces on average, "
            f"up to {max(fragment_pieces)}"
        )

    print(f"\n  why {unmatched} predictions matched nothing")
    for category in PREDICTION_CATEGORIES:
        count = prediction_counts[category]
        print(f"    {category:<12} {count:>6}   {count / max(unmatched, 1):>6.1%}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
