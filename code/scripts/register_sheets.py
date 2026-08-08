#!/usr/bin/env python
"""Validate a capture session and align every scan to its digital crease pattern.

Run this after each folding/scanning session -- not at the end of the project.
It reports manifest problems while the sheet is still on your desk, then writes
one overlay image per scan so you can eyeball the alignment.

    python scripts/register_sheets.py data/manifest.csv \
        --images data/scans --designs data/designs --out data/overlays

Sheets flagged NOT CONFIDENT are not necessarily wrong; they are the ones worth
opening. Four-fold symmetric patterns are legitimately ambiguous and will always
land there.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from origamicp.capture import BACK, Manifest, back_face, overlay, register
from origamicp.core import CreasePattern
from origamicp.verify import verify


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="capture manifest CSV")
    parser.add_argument("--images", type=Path, default=Path("."), help="root for image_path")
    parser.add_argument("--designs", type=Path, default=Path("designs"), help="dir of <design_id>.fold")
    parser.add_argument("--out", type=Path, default=None, help="where to write overlays")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest = Manifest.load(args.manifest)
    print(manifest.summary())

    issues = manifest.validate(args.images)
    errors = [i for i in issues if i.severity == "error"]
    for issue in issues:
        print(f"  {issue}")
    if errors:
        print(f"\n{len(errors)} error(s) -- fix these before the sheets are filed away.")
    if args.validate_only:
        return 1 if errors else 0
    if errors:
        return 1

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    designs: dict[str, CreasePattern] = {}
    rows, low_confidence = [], []

    for record in manifest:
        if record.design_id not in designs:
            path = args.designs / f"{record.design_id}.fold"
            if not path.exists():
                print(f"  [error] {record.sheet_id}: no design file {path}")
                return 1
            cp = CreasePattern.from_fold(path)
            report = verify(cp)
            if not report.valid:
                # The reference pattern itself should fold; if it does not, the
                # sheet was folded from a broken template.
                print(f"  [warning] design {record.design_id} is not locally valid: {report.summary()}")
            designs[record.design_id] = cp

        cp = designs[record.design_id]
        # The back scan sees a mirrored sheet with every M and V exchanged.
        truth = back_face(cp) if record.face == BACK else cp

        image = cv2.imread(str(args.images / record.image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  [error] unreadable image: {record.image_path}")
            return 1

        result = register(image, truth)
        rows.append((record, result))
        if not result.confident:
            low_confidence.append(record)

        if args.out:
            name = Path(record.image_path).stem
            cv2.imwrite(str(args.out / f"{name}_overlay.jpg"), overlay(image, truth, result))

    print(f"\n{'sheet':<12}{'face':<7}{'rot':<5}{'score':>8}{'margin':>9}  status")
    for record, result in rows:
        status = "ok" if result.confident else "CHECK"
        print(
            f"{record.sheet_id:<12}{record.face:<7}{record.rotation_deg:<5}"
            f"{result.score:>8.3f}{result.margin:>9.2f}  {status}"
        )

    scores = np.array([r.score for _, r in rows])
    print(f"\nmedian alignment score: {np.median(scores):.3f}")
    print(f"needs a look: {len(low_confidence)}/{len(rows)}")
    if args.out:
        print(f"overlays written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
