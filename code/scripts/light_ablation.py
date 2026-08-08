#!/usr/bin/env python
"""Is mountain/valley identifiable from one image? The light-direction ablation.

    python scripts/light_ablation.py --data ../data/synth \
        --with-light ../runs/baseline/best.pt --without-light ../runs/nolight/best.pt

A mountain lit from the left and a valley lit from the right are the same
picture. So under a single light the assignment is fixed only up to inverting
every crease at once, and Maekawa's condition cannot break the tie either --
swapping all of them leaves the defect at plus or minus two.

Plain accuracy cannot tell "learned nothing" apart from "learned everything
except the sign", so we report two numbers per sheet:

  accuracy            fraction of creases labelled correctly
  accuracy-up-to-flip max(accuracy, 1 - accuracy)

A model that has not learned the structure sits near 0.5 on both. A model that
has learned it but cannot orient it would sit near 0.5 on the first and near 1.0
on the second, with per-sheet accuracies piled up at 0 and 1.

Measured, the ablated model does neither: 0.535 plain, 0.734 up to a flip, and
per-sheet accuracies spread across the whole range rather than bunched at the
ends. So the global flip is not the only thing it is missing. The theory says
the assignment is recoverable up to one sign for the sheet as a whole, but
choosing one sign and applying it everywhere requires creases to be compared
with each other, and a convolutional network given no light reference has no
local evidence to compare. It ends up guessing per crease, only partly
consistently -- 0.696 sits between the 0.5 of independent guesses and the 1.0
the theory allows.

That gap is the argument for decoding under the fold theorems rather than
per pixel: Maekawa's condition constrains creases at a vertex jointly, which is
exactly the global consistency the architecture fails to supply on its own.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--with-light", type=Path, required=True)
    parser.add_argument("--without-light", type=Path, required=True)
    parser.add_argument("--split", default="test")
    # Matches every other script; the earlier 0.85 predated the
    # validation sweep and left this one stage quoting a different mask.
    parser.add_argument("--threshold", type=float, default=0.92)
    parser.add_argument("--out", type=Path, default=Path("light_ablation.png"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def per_sheet_accuracy(checkpoint_path, data, split, threshold, device):
    """MV accuracy on each sheet, scored only where a crease was found."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    use_light = checkpoint.get("use_light", True)
    model = CreaseUNet(
        in_channels=3,
        width=checkpoint.get("width", 32),
        depth=checkpoint.get("depth", 4),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = SyntheticCreaseDataset(
        data, split, crop=None, jitter=False, use_light=use_light
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate)

    accuracies = []
    with torch.no_grad():
        for batch in loader:
            if float(batch["mv_valid"][0]) == 0.0:
                continue  # back-lit sheets carry no direction at all
            tensors = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            outputs = model(tensors["image"])

            sheet = tensors["sheet"].bool()
            truth = tensors["crease"].bool() & sheet
            found = (outputs["crease"].sigmoid() > threshold) & truth
            if not found.any():
                continue

            predicted = outputs["mv"].argmax(dim=1)
            target = tensors["mv"].clamp(min=1) - 1
            accuracies.append(float(((predicted == target) & found).sum() / found.sum()))

    return np.array(accuracies), use_light


def independent_flip_reference(data, split, trials=4000, seed=0):
    """Up-to-flip accuracy of a model that decides each crease by a fair coin.

    The ambiguity leaves exactly one bit of uncertainty for a whole sheet, so a
    predictor that resolved it coherently would score 1.0 up to a flip. That
    ceiling alone does not say whether a measured 0.73 is close to coherent or
    close to nothing, because a model flipping every crease independently
    already beats one half: with n creases the better of the two global signs
    wins by the fluctuation of a binomial, roughly 0.4/sqrt(n).

    This computes that floor on the actual crease-count distribution rather than
    the asymptotic form, which matters at the small n a single sheet has.

    Creases are weighted by length, since the measured accuracy is over crease
    pixels and a crease contributes pixels in proportion to how long it is.
    """
    dataset = SyntheticCreaseDataset(data, split, crop=None, jitter=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate)
    rng = np.random.default_rng(seed)

    expected = []
    for index, batch in enumerate(loader):
        if float(batch["mv_valid"][0]) == 0.0:
            continue
        cp, corners = truth_pattern(dataset, dataset.rows[index])
        segments = crease_segments(cp, project_to_pixels(cp, corners))
        if not segments:
            continue
        weights = np.array([float(np.linalg.norm(s[1] - s[0])) for s in segments])
        total = weights.sum()
        if total <= 0:
            continue
        draws = rng.random((trials, len(weights))) < 0.5
        accuracy = draws @ weights / total
        expected.append(float(np.maximum(accuracy, 1.0 - accuracy).mean()))

    return np.array(expected)


def report(name, accuracies):
    flipped = np.maximum(accuracies, 1.0 - accuracies)
    extreme = np.mean((accuracies < 0.1) | (accuracies > 0.9))
    print(f"\n{name}  ({len(accuracies)} sheets)")
    print(f"  MV accuracy               {accuracies.mean():.3f} +- {accuracies.std():.3f}")
    print(f"  MV accuracy up to a flip  {flipped.mean():.3f} +- {flipped.std():.3f}")
    print(f"  sheets that are all-right or all-wrong  {100 * extreme:.1f}%")
    return flipped


def main(argv=None) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args(argv)
    device = torch.device(args.device)

    with_light, flag_a = per_sheet_accuracy(
        args.with_light, args.data, args.split, args.threshold, device
    )
    without_light, flag_b = per_sheet_accuracy(
        args.without_light, args.data, args.split, args.threshold, device
    )
    if not flag_a or flag_b:
        print(
            f"warning: checkpoints report use_light={flag_a} and {flag_b}; "
            "expected True then False"
        )

    report("with light direction", with_light)
    flipped = report("without light direction", without_light)

    # Three numbers make the claim readable where two do not: what independent
    # guessing already gets, what the ablated model gets, and what resolving the
    # single bit of ambiguity would get.
    reference = independent_flip_reference(args.data, args.split)
    if len(reference):
        print(
            f"\nup-to-flip accuracy, three reference points"
            f"\n  creases flipped independently  {reference.mean():.3f}"
            f"\n  ablated model                  {flipped.mean():.3f}"
            f"\n  one bit resolved coherently    1.000"
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    bins = np.linspace(0, 1, 26)
    for ax, values, title in [
        (axes[0], with_light, "light direction given"),
        (axes[1], without_light, "light direction withheld"),
    ]:
        ax.hist(values, bins=bins, color="#3b6ea5", edgecolor="white")
        ax.axvline(0.5, color="#c1272d", ls="--", lw=1.5, label="chance")
        ax.set_title(f"{title}\nmean accuracy {values.mean():.3f}", fontsize=11)
        ax.set_xlabel("per-sheet mountain/valley accuracy")
        ax.set_ylabel("sheets")
        ax.legend(fontsize=9)
    fig.suptitle(
        "Without the light direction the assignment collapses to chance -- and not "
        "merely by a global flip:\nallowing the best flip per sheet recovers only "
        f"{flipped.mean():.3f}, against the 1.0 the ambiguity alone would permit.",
        fontsize=11.5,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nfigure -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
