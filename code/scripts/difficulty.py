#!/usr/bin/env python
"""How performance depends on how visible the crease actually is.

    python scripts/difficulty.py --data ../data/synth --checkpoint ../runs/baseline/best.pt

Difficulty is measured on each image rather than read off the parameters that
generated it: the same settings produce a range of images, and a curve binned by
the intended difficulty rather than the realised one washes out.

The x axis is the crease signal-to-texture ratio -- the brightness step across a
crease divided by the same measurement taken on blank paper. It has a meaning
outside this codebase, which is what makes the curve usable as a calibration
target: measure it on real scans and the plot says what to expect from them.

Read the mountain/valley curve with care. It is scored only where a crease was
actually detected, so at low visibility it is scored on whichever creases were
clear enough to find -- a self-selected easier subset. The near-flat curve
therefore understates how much the MV call really suffers on faint creases; the
detection curve carries that cost instead. Scoring MV on missed creases would
just double-count detection failures, so the conditioning is deliberate, but it
has to be stated rather than read as "MV is robust".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from origamicp.capture import back_face
from origamicp.core.cp import CreasePattern
from origamicp.models import CreaseUNet, SyntheticCreaseDataset, collate
from origamicp.render import crease_snr_of_image


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--bins", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("difficulty.png"))
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

    dataset = SyntheticCreaseDataset(
        args.data, args.split, crop=None, jitter=False,
        use_light=checkpoint.get("use_light", True),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate)

    snrs, f1s, mvs = [], [], []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            row = dataset.rows[index]
            if float(batch["mv_valid"][0]) == 0.0:
                continue

            cp = CreasePattern.from_fold(dataset.root / row["fold_path"])
            if row["face"] == "back":
                cp = back_face(cp)
            corners = np.fromstring(row["corners"], sep=" ").reshape(4, 2)
            gray = (batch["image"][0, 0].numpy() * 255).astype(np.uint8)
            signal, noise = crease_snr_of_image(gray, cp, corners)
            if noise <= 0:
                continue

            tensors = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            outputs = model(tensors["image"])
            sheet = tensors["sheet"].bool()
            truth = tensors["crease"].bool() & sheet
            predicted = (outputs["crease"].sigmoid() > args.threshold) & sheet

            tp = float((predicted & truth).sum())
            fp = float((predicted & ~truth).sum())
            fn = float((~predicted & truth).sum())
            if tp + fp == 0 or tp + fn == 0:
                continue
            precision, recall = tp / (tp + fp), tp / (tp + fn)
            if precision + recall == 0:
                continue

            found = predicted & truth
            if not found.any():
                continue
            mv_predicted = outputs["mv"].argmax(dim=1)
            mv_target = tensors["mv"].clamp(min=1) - 1

            snrs.append(signal / noise)
            f1s.append(2 * precision * recall / (precision + recall))
            mvs.append(float(((mv_predicted == mv_target) & found).sum() / found.sum()))

    snrs, f1s, mvs = np.array(snrs), np.array(f1s), np.array(mvs)
    edges = np.quantile(snrs, np.linspace(0, 1, args.bins + 1))
    edges[-1] += 1e-9

    print(f"\n{len(snrs)} sheets, crease signal-to-texture {snrs.min():.2f} to {snrs.max():.2f}\n")
    print(f"{'SNR range':>16}{'n':>6}{'crease F1':>12}{'MV accuracy':>14}")
    centres, binned_f1, binned_mv = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (snrs >= lo) & (snrs < hi)
        if mask.sum() < 3:
            continue
        centres.append(float(np.median(snrs[mask])))
        binned_f1.append(float(f1s[mask].mean()))
        binned_mv.append(float(mvs[mask].mean()))
        print(f"{lo:>7.2f} - {hi:<6.2f}{int(mask.sum()):>6}{binned_f1[-1]:>12.3f}{binned_mv[-1]:>14.3f}")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(centres, binned_f1, "o-", color="#3b6ea5", lw=2, label="crease detection F1")
    ax.plot(centres, binned_mv, "s-", color="#c1272d", lw=2, label="mountain/valley accuracy")
    ax.axhline(0.5, color="#999", ls="--", lw=1.2)
    ax.text(centres[-1], 0.515, "MV chance", ha="right", fontsize=9, color="#777")
    ax.set_xlabel("crease signal-to-texture ratio (measured on the image)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)
    ax.set_title(
        "Detection follows crease visibility; the mountain/valley call barely does\n"
        "(MV is scored only where a crease was found -- see the note in this script)",
        fontsize=11.5,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nfigure -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
