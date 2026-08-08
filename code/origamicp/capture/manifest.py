"""Capture manifest: one row per scan, plus the checks that keep a session honest.

Data collection runs for weeks. A missing back-face scan or an accidental JPEG
is cheap to fix the same evening and expensive to fix a month later, once the
sheet has been unfolded, re-flattened and filed away. So the validator runs
against the manifest after every session, not once at the end.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

FRONT, BACK = "front", "back"
SCANNER, PHOTOMETRIC, BACKLIT = "scanner", "photometric", "backlit"

# JPEG's ringing sits at the same spatial frequency as the shading that
# distinguishes a mountain from a valley, so it is rejected outright.
LOSSLESS_SUFFIXES = frozenset({".png", ".tif", ".tiff"})
MIN_DPI = 600

# Two faces at two rotations: the faces give MV ground truth, the rotations give
# a second light azimuth so creases parallel to the scanner's lamp still show up.
REQUIRED_SCANNER_VIEWS = frozenset({(FRONT, 0), (FRONT, 90), (BACK, 0), (BACK, 90)})


@dataclass
class CaptureRecord:
    sheet_id: str
    design_id: str
    image_path: str
    modality: str = SCANNER
    face: str = FRONT
    rotation_deg: int = 0
    dpi: int = 600
    paper_type: str = ""
    gsm: int = 0
    color: str = "white"
    light_azimuth_deg: float = -1.0  # -1 means "not applicable" (scanner/backlit)
    folder_name: str = ""
    fold_date: str = ""
    split: str = ""
    notes: str = ""

    @property
    def view(self) -> tuple[str, int]:
        return (self.face, int(self.rotation_deg))


@dataclass(frozen=True)
class Issue:
    severity: str  # "error" blocks training; "warning" is worth a look
    sheet_id: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.sheet_id}: {self.message}"


class Manifest:
    """A list of capture records with load/save and validation."""

    COLUMNS = [f.name for f in fields(CaptureRecord)]

    def __init__(self, records: list[CaptureRecord] | None = None) -> None:
        self.records: list[CaptureRecord] = list(records or [])

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def add(self, record: CaptureRecord) -> None:
        self.records.append(record)

    def by_sheet(self) -> dict[str, list[CaptureRecord]]:
        grouped: dict[str, list[CaptureRecord]] = defaultdict(list)
        for r in self.records:
            grouped[r.sheet_id].append(r)
        return dict(grouped)

    def by_design(self) -> dict[str, list[CaptureRecord]]:
        grouped: dict[str, list[CaptureRecord]] = defaultdict(list)
        for r in self.records:
            grouped[r.design_id].append(r)
        return dict(grouped)

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        types = {f.name: f.type for f in fields(CaptureRecord)}
        records = []
        with Path(path).open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                clean = {}
                for key, raw in row.items():
                    if key not in types:
                        continue
                    if raw == "":
                        continue
                    clean[key] = {"int": int, "float": float}.get(types[key], str)(raw)
                records.append(CaptureRecord(**clean))
        return cls(records)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.COLUMNS)
            writer.writeheader()
            for r in self.records:
                writer.writerow(asdict(r))

    def validate(self, root: str | Path = ".") -> list[Issue]:
        """Check completeness, file hygiene and split integrity.

        ``root`` is the directory ``image_path`` values are relative to.
        """
        root = Path(root)
        issues: list[Issue] = []

        seen_paths: dict[str, str] = {}
        for r in self.records:
            path = root / r.image_path

            if r.image_path in seen_paths and seen_paths[r.image_path] != r.sheet_id:
                issues.append(
                    Issue("error", r.sheet_id, f"image reused by {seen_paths[r.image_path]}: {r.image_path}")
                )
            seen_paths.setdefault(r.image_path, r.sheet_id)

            if not path.exists():
                issues.append(Issue("error", r.sheet_id, f"missing image: {r.image_path}"))
            if path.suffix.lower() not in LOSSLESS_SUFFIXES:
                issues.append(
                    Issue("error", r.sheet_id, f"lossy or unknown format: {path.suffix} (re-scan as PNG/TIFF)")
                )
            if r.dpi < MIN_DPI:
                issues.append(Issue("error", r.sheet_id, f"dpi {r.dpi} below {MIN_DPI}"))
            if r.face not in (FRONT, BACK):
                issues.append(Issue("error", r.sheet_id, f"unknown face: {r.face!r}"))
            if not r.design_id:
                issues.append(Issue("error", r.sheet_id, "empty design_id"))
            if r.modality == PHOTOMETRIC and r.light_azimuth_deg < 0:
                issues.append(
                    Issue("error", r.sheet_id, "photometric capture needs light_azimuth_deg")
                )
            if not r.folder_name:
                issues.append(Issue("warning", r.sheet_id, "no folder_name recorded"))

        for sheet_id, rows in self.by_sheet().items():
            scanner_views = {r.view for r in rows if r.modality == SCANNER}
            if scanner_views:
                for missing in sorted(REQUIRED_SCANNER_VIEWS - scanner_views):
                    issues.append(
                        Issue("error", sheet_id, f"missing scan: face={missing[0]} rotation={missing[1]}")
                    )
            if len({r.design_id for r in rows}) > 1:
                issues.append(Issue("error", sheet_id, "sheet maps to more than one design_id"))

            azimuths = [r.light_azimuth_deg for r in rows if r.modality == PHOTOMETRIC]
            if azimuths and len(set(azimuths)) < 4:
                issues.append(
                    Issue("warning", sheet_id, f"only {len(set(azimuths))} light azimuths; normals will be poorly conditioned")
                )

        # The failure that quietly inflates results: the same design in two
        # splits means the test set measures memorisation, not perception.
        for design_id, rows in self.by_design().items():
            splits = {r.split for r in rows if r.split}
            if len(splits) > 1:
                issues.append(
                    Issue("error", design_id, f"design leaks across splits: {sorted(splits)}")
                )

        return issues

    def summary(self) -> str:
        by_modality: dict[str, int] = defaultdict(int)
        for r in self.records:
            by_modality[r.modality] += 1
        parts = ", ".join(f"{k}={v}" for k, v in sorted(by_modality.items()))
        return (
            f"{len(self.records)} scans | {len(self.by_sheet())} sheets | "
            f"{len(self.by_design())} designs | {parts}"
        )
