#!/usr/bin/env python
"""What a model does with the one bit the image does not contain.

    python scripts/coherence.py --data ../data/synth768 \
        ../runs/nolight768/best.pt ../runs/nl_full/best.pt ../runs/nl_flipinv/best.pt

Without the light direction a sheet's mountain/valley assignment is fixed only
up to inverting every crease at once. That is one bit for the whole pattern, so
plain accuracy is the wrong question -- it is bounded near one half no matter
how well the pattern was read. Three columns take its place:

  up to a flip   the better of the two global signs, per sheet
  one crease     the true label of a single crease, used to pick the sign
  coherent       sheets that come back all-right or all-wrong

``one crease`` is the practically interesting one. The theory says a single bit
resolves the entire sheet; this measures whether a given model has actually
learned the relative structure that would let it. The reference crease is the
longest in the ground-truth pattern, scored by majority over its pixels, which
is what observing one fold would give you.

The independent-flip row is the floor. A model deciding every crease by a fair
coin still scores well above one half up to a flip, because the better of the
two signs wins by a binomial fluctuation, so a raw up-to-flip number cannot be
read without it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from origamicp.models import (
    CreaseUNet,
    SyntheticCreaseDataset,
    collate,
    project_to_pixels,
)
from origamicp.vectorize.match import crease_segments
from scripts.evaluate_graph import truth_pattern
from scripts.light_ablation import independent_flip_reference


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.92)
    parser.add_argument("--out", type=Path, help="write the per-sheet accuracy figure here")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def _reference_crease(cp, corners, shape, thickness=5):
    """A mask over the longest ground-truth crease: the one fold you get to see."""
    segments = crease_segments(cp, project_to_pixels(cp, corners))
    if not segments:
        return None
    longest = max(segments, key=lambda s: float(np.linalg.norm(s[1] - s[0])))
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.line(
        mask,
        tuple(np.round(longest[0]).astype(int)),
        tuple(np.round(longest[1]).astype(int)),
        1,
        thickness,
    )
    return mask.astype(bool)


def evaluate(checkpoint_path, data, split, threshold, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    use_light = checkpoint.get("use_light", True)
    model = CreaseUNet(
        in_channels=3,
        width=checkpoint.get("width", 32),
        depth=checkpoint.get("depth", 4),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = SyntheticCreaseDataset(data, split, crop=None, jitter=False, use_light=use_light)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate)

    plain, resolved = [], []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if float(batch["mv_valid"][0]) == 0.0:
                continue
            tensors = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            outputs = model(tensors["image"])

            truth = tensors["crease"].bool() & tensors["sheet"].bool()
            found = (outputs["crease"].sigmoid() > threshold) & truth
            if not found.any():
                continue

            correct = (outputs["mv"].argmax(dim=1) == (tensors["mv"].clamp(min=1) - 1)) & found
            accuracy = float(correct.sum() / found.sum())
            plain.append(accuracy)

            cp, corners = truth_pattern(dataset, dataset.rows[index])
            reference = _reference_crease(cp, corners, found.shape[-2:])
            if reference is None:
                resolved.append(max(accuracy, 1.0 - accuracy))
                continue

            on_reference = found[0].cpu().numpy() & reference
            if not on_reference.any():
                # No detected pixels on the crease we were allowed to look at, so
                # the bit is unavailable and the sheet keeps whatever sign it has.
                resolved.append(accuracy)
                continue

            # Majority vote over the reference crease decides whether to invert.
            agrees = float(correct[0].cpu().numpy()[on_reference].mean()) >= 0.5
            resolved.append(accuracy if agrees else 1.0 - accuracy)

    return np.array(plain), np.array(resolved)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)

    reference = independent_flip_reference(args.data, args.split)
    rows = [("creases flipped independently", 0.5, float(reference.mean()), np.nan, np.nan)]
    distributions = []

    for path in args.checkpoints:
        plain, resolved = evaluate(path, args.data, args.split, args.threshold, device)
        if not len(plain):
            print(f"{path}: nothing scored")
            continue
        flipped = np.maximum(plain, 1.0 - plain)
        coherent = float(np.mean((plain < 0.1) | (plain > 0.9)))
        rows.append(
            (path.parent.name, float(plain.mean()), float(flipped.mean()),
             float(resolved.mean()), coherent)
        )
        distributions.append((path.parent.name, plain))

    print(f"\nmountain/valley without the light direction, {args.split} split\n")
    print(f"  {'model':<18}{'plain':>9}{'up to flip':>12}{'one crease':>12}{'coherent':>10}")
    for name, plain_mean, flip_mean, resolved_mean, coherent in rows:
        resolved_text = "     --" if np.isnan(resolved_mean) else f"{resolved_mean:>12.3f}"
        coherent_text = "      --" if np.isnan(coherent) else f"{coherent:>9.1%}"
        print(f"  {name:<18}{plain_mean:>9.3f}{flip_mean:>12.3f}{resolved_text}{coherent_text}")
    print("\n  one bit resolved coherently would be 1.000 in every column but the first")

    if args.out is not None and distributions:
        _figure(distributions, args.out)
        print(f"\nfigure -> {args.out}")
    return 0


def _figure(distributions, out):
    """Per-sheet accuracy histograms, which show the shape the table cannot.

    A model that has learned the relative structure but not the global sign puts
    every sheet near zero or near one -- it decides, and is wholly right or
    wholly wrong. A model minimising a per-pixel loss puts them in a heap at one
    half, which is the same mean and a different thing entirely.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(distributions), figsize=(3.1 * len(distributions), 2.9), sharey=True
    )
    if len(distributions) == 1:
        axes = [axes]

    bins = np.linspace(0, 1, 26)
    for ax, (name, values) in zip(axes, distributions):
        ax.hist(values, bins=bins, color="#3b6ea5", edgecolor="white", linewidth=0.5)
        ax.axvline(0.5, color="#999999", lw=1, ls=":")
        ax.set_title(f"{name}\nmean {values.mean():.3f}", fontsize=10)
        ax.set_xlabel("per-sheet MV accuracy")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("sheets")
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    sys.exit(main())
