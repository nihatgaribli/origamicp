#!/usr/bin/env python
"""Write the pilot design set: FOLD ground truth plus printable templates.

    python scripts/make_pilot.py --out ../data

Two files come out per design, and the difference between them is the part that
is easy to get backwards:

  designs/<id>.fold        ground truth in FRONT-view coordinates. This is what
                           register_sheets.py compares a front scan against.
  templates/<id>_print.svg what you actually print, on the BACK of the sheet.
                           It is the mirrored, MV-swapped pattern, because the
                           folder is looking at the reverse side. Fold every
                           line as the template marks it and the finished
                           sheet's front face matches the .fold exactly.

Printing the front-view pattern by mistake mirrors every sheet in the dataset,
which registration would silently absorb -- the MV labels would just all be
wrong. Hence the two names.

Print at 100% / "actual size", never "fit to page": the SVG is sized in
millimetres and the geometry is only ground truth if the scale is exact.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from origamicp.capture import back_face
from origamicp.generate import chamfer
from origamicp.generate.designs import PILOT_DESIGNS, SHEET_MM
from origamicp.render import to_svg
from origamicp.verify import verify


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--sheet-mm", type=float, default=SHEET_MM)
    parser.add_argument(
        "--chamfer-mm",
        type=float,
        default=8.0,
        help="corner cut that fixes sheet orientation; 0 disables it",
    )
    parser.add_argument(
        "--front-reference",
        action="store_true",
        help="also write front-view SVGs (for reading, not for printing)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    designs_dir = args.out / "designs"
    templates_dir = args.out / "templates"

    print(f"{'design':<18}{'creases':>8}{'interior':>10}  status")
    failed = []

    for name, make in PILOT_DESIGNS.items():
        cp = make()
        if args.chamfer_mm > 0:
            # Cut before writing the FOLD file: the chamfer is part of the
            # sheet's geometry, so ground truth and template must both carry
            # it or registration will look for a corner that is not there.
            cp = chamfer(cp, args.chamfer_mm)
        report = verify(cp)
        n_interior = len(report.interior)

        if not report.valid:
            failed.append(name)
            print(f"{name:<18}{cp.n_edges:>8}{n_interior:>10}  REJECTED  {report.summary()}")
            continue

        cp.to_fold(designs_dir / f"{name}.fold", name=name)
        to_svg(
            back_face(cp),
            templates_dir / f"{name}_print.svg",
            label=(
                f"{name}  |  PRINT ON BACK  |  100% scale  |  {args.sheet_mm:.0f}mm"
                + ("  |  CUT THE CLIPPED CORNER" if args.chamfer_mm > 0 else "")
            ),
            sheet_mm=args.sheet_mm,
        )
        if args.front_reference:
            to_svg(
                cp,
                templates_dir / f"{name}_front_reference.svg",
                label=f"{name}  |  front view -- reference only, do not print to fold",
                sheet_mm=args.sheet_mm,
            )
        print(f"{name:<18}{cp.n_edges:>8}{n_interior:>10}  ok")

    if failed:
        print(f"\n{len(failed)} design(s) rejected; nothing written for them.")
        return 1

    print(f"\nFOLD ground truth  -> {designs_dir}")
    print(f"printable templates -> {templates_dir}")
    print("\nPrint the *_print.svg files at 100% scale onto the back of each sheet.")
    if args.chamfer_mm > 0:
        print(
            f"Cut along the grey outline, including the {args.chamfer_mm:.0f}mm clipped "
            "corner.\nThat cut is the only thing telling registration which way round the "
            "sheet was scanned:\nwithout it a symmetric pattern registers upside down and "
            "every M and V comes back swapped."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
