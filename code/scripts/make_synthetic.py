#!/usr/bin/env python
"""Generate the synthetic pre-training set.

    python scripts/make_synthetic.py --n 5000 --out ../data/synth

Each sample is a randomly drawn foldable pattern rendered as a scan of paper,
written with its FOLD ground truth. Both faces of every sheet are rendered: the
back is the mirrored, MV-swapped pattern under the same physics, which is what
makes the flip-consistency diagnostic trainable rather than only measurable.

Splits are assigned by design, never by image, for the same reason they are in
the real manifest -- a design appearing in two splits turns the test set into a
memorisation check.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from origamicp.capture import back_face
from origamicp.render import BACKLIT, PHOTOMETRIC, SCANNER, ScanStyle, render
from origamicp.generate.random_cp import random_design
from origamicp.verify import verify

COLUMNS = [
    "sample_id", "design_id", "family", "face", "image_path", "fold_path",
    "modality", "light_azimuth_deg", "light_elevation_deg", "rotation_deg",
    "crease_height", "undulation", "fiber_noise", "size_px", "corners", "split",
]


def sample_style(rng: np.random.Generator, size_px: int) -> ScanStyle:
    """Randomise capture conditions over the range the protocol can produce."""
    mode = str(rng.choice([SCANNER, SCANNER, SCANNER, PHOTOMETRIC, BACKLIT],
                          p=[0.25, 0.25, 0.2, 0.2, 0.1]))
    return ScanStyle(
        size_px=size_px,
        mode=mode,
        light_azimuth_deg=float(rng.uniform(0, 360)),
        light_elevation_deg=float(rng.uniform(12, 32)),
        # These three set how visible the MV cue is. The range spans a crease
        # signal-to-texture ratio of about 1.3 to 5 (see measure_crease_snr), so
        # the set runs from obvious to genuinely hard rather than sitting at one
        # difficulty. Recorded per sample, so accuracy can be reported against
        # it instead of averaged over an unknown mixture.
        crease_height=float(rng.uniform(0.50, 1.50)),
        undulation=float(rng.uniform(0.20, 1.00)),
        fiber_noise=float(rng.uniform(0.020, 0.070)),
        crease_width_px=float(rng.uniform(1.2, 2.4)),
        crease_wobble_px=float(rng.uniform(0.2, 1.1)),
        paper_level=float(rng.uniform(0.72, 0.90)),
        ambient=float(rng.uniform(0.45, 0.68)),
        sensor_noise=float(rng.uniform(0.002, 0.010)),
        perspective=0.0 if mode != PHOTOMETRIC else float(rng.uniform(0.0, 0.02)),
        rotation_deg=None,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=1000, help="number of designs")
    parser.add_argument("--out", type=Path, default=Path("data/synth"))
    parser.add_argument("--size-px", type=int, default=768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    images_dir = args.out / "images"
    folds_dir = args.out / "folds"
    images_dir.mkdir(parents=True, exist_ok=True)
    folds_dir.mkdir(parents=True, exist_ok=True)

    rows, families = [], {}
    for i in range(args.n):
        family, cp = random_design(rng)
        report = verify(cp)
        if not report.valid:
            # Should be unreachable: every family is foldable by construction.
            print(f"  [error] {family} produced an invalid pattern; aborting")
            return 1

        design_id = f"{family}_{i:06d}"
        cp.to_fold(folds_dir / f"{design_id}.fold", name=design_id)
        families[family] = families.get(family, 0) + 1

        draw = rng.random()
        split = "test" if draw < args.test_frac else (
            "val" if draw < args.test_frac + args.val_frac else "train"
        )

        style = sample_style(rng, args.size_px)
        for face, truth in (("front", cp), ("back", back_face(cp))):
            # Same style, same seed: only the sheet is turned over.
            image, corners = render(
                truth, style, np.random.default_rng(args.seed * 7919 + i)
            )
            name = f"{design_id}_{face}.png"
            cv2.imwrite(str(images_dir / name), image)
            rows.append({
                "sample_id": f"{design_id}_{face}",
                "design_id": design_id,
                "family": family,
                "face": face,
                "image_path": f"images/{name}",
                "fold_path": f"folds/{design_id}.fold",
                "modality": style.mode,
                "light_azimuth_deg": round(style.light_azimuth_deg, 2),
                "light_elevation_deg": round(style.light_elevation_deg, 2),
                "rotation_deg": "",
                "crease_height": round(style.crease_height, 3),
                "undulation": round(style.undulation, 3),
                "fiber_noise": round(style.fiber_noise, 4),
                "size_px": style.size_px,
                # Without the pose the image and the FOLD file cannot be put
                # back into correspondence, and no pixel-level target can be
                # rebuilt from them.
                "corners": " ".join(f"{v:.3f}" for v in corners.ravel()),
                "split": split,
            })

        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{args.n} designs")

    index = args.out / "index.csv"
    with index.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    print(f"\n{len(rows)} images from {args.n} designs -> {args.out}")
    print("  families:", ", ".join(f"{k}={v}" for k, v in sorted(families.items())))
    print("  splits:  ", ", ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"  index:    {index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
