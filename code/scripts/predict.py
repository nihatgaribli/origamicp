#!/usr/bin/env python
"""Run a trained extractor over held-out sheets and write a comparison figure.

    python scripts/predict.py --data ../data/synth --checkpoint ../runs/baseline/best.pt

Prediction and ground truth are shown side by side rather than as one blended
overlay: a mountain drawn where a valley belongs has to be visible as a colour
disagreement, and blending hides exactly that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from origamicp.models import Counts, CreaseUNet, SyntheticCreaseDataset, collate


def colourise(crease: np.ndarray, mv: np.ndarray, base: np.ndarray) -> np.ndarray:
    rgb = np.stack([base] * 3, axis=-1)
    rgb[(crease > 0) & (mv == 0)] = [1.0, 0.15, 0.15]
    rgb[(crease > 0) & (mv == 1)] = [0.1, 0.45, 1.0]
    return rgb


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=Path("predictions.png"))
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    counts = Counts()
    shown = []
    with torch.no_grad():
        for batch in loader:
            tensors = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            outputs = model(tensors["image"])
            counts.update(outputs, tensors, args.threshold)

            if len(shown) < args.n and float(tensors["mv_valid"][0]) > 0:
                base = batch["image"][0, 0].numpy()
                predicted_crease = (
                    outputs["crease"][0].sigmoid() > args.threshold
                ).float().cpu().numpy() * batch["sheet"][0].numpy()
                shown.append(
                    (
                        base,
                        colourise(
                            batch["crease"][0].numpy(),
                            (batch["mv"][0].clamp(min=1) - 1).numpy(),
                            base,
                        ),
                        colourise(
                            predicted_crease,
                            outputs["mv"][0].argmax(0).cpu().numpy(),
                            base,
                        ),
                    )
                )

    print(f"{args.split}: {counts.summary()}")

    rows = len(shown)
    fig, axes = plt.subplots(rows, 3, figsize=(11, 3.7 * rows), squeeze=False)
    for row, (base, truth, predicted) in enumerate(shown):
        for col, (image, title) in enumerate(
            [(base, "input scan"), (truth, "ground truth"), (predicted, "prediction")]
        ):
            axes[row][col].imshow(image, cmap="gray" if col == 0 else None, vmin=0, vmax=1)
            axes[row][col].axis("off")
            if row == 0:
                axes[row][col].set_title(title, fontsize=11)
    fig.suptitle(
        f"Held-out sheets  |  crease F1 {counts.f1:.3f}  |  MV accuracy "
        f"{counts.mv_accuracy:.3f}  (chance 0.5)  |  red = mountain, blue = valley",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=100, bbox_inches="tight")
    print(f"figure -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
