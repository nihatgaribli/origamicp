"""Torch dataset over a generated synthetic set.

No geometric augmentation here, deliberately. The renderer already samples the
sheet's rotation uniformly, so flips and rotations would add little -- and each
one would have to carry the light azimuth along with it, since turning the image
turns the light. That bookkeeping is a silent-corruption risk for no gain, so
augmentation is limited to photometric jitter, which leaves the labels alone.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from origamicp.core.cp import CreasePattern
from origamicp.models.targets import build_targets, light_channels
from origamicp.render.scan import BACKLIT


class SyntheticCreaseDataset(Dataset):
    """Images with crease, mountain/valley and sheet targets.

    Each item is a dict of tensors:
      ``image``     (3, H, W) -- greyscale scan plus the two light planes
      ``crease``    (H, W) float, 1 on a fold
      ``mv``        (H, W) long, 0 background / 1 mountain / 2 valley
      ``sheet``     (H, W) float, 1 inside the paper
      ``mv_valid``  scalar float, 0 for back-lit sheets

    ``mv_valid`` is what keeps back-lit samples honest. Transmission lights both
    fold directions identically, so their MV labels are unlearnable; masking
    them out of the MV loss keeps the crease-detection signal, which is the one
    thing back-lighting is good at.
    """

    def __init__(
        self,
        root: str | Path,
        split: str | None = "train",
        crop: int | None = 512,
        jitter: bool = True,
        seed: int = 0,
        use_light: bool = True,
    ) -> None:
        self.root = Path(root)
        self.crop = crop
        self.jitter = jitter
        self.use_light = use_light
        self.rng = np.random.default_rng(seed)

        with (self.root / "index.csv").open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.rows = [r for r in rows if split is None or r["split"] == split]
        if not self.rows:
            raise ValueError(f"no rows for split {split!r} in {self.root}")
        self._patterns: dict[str, CreasePattern] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _pattern(self, row: dict) -> CreasePattern:
        key = row["fold_path"]
        if key not in self._patterns:
            self._patterns[key] = CreasePattern.from_fold(self.root / key)
        return self._patterns[key]

    def _crop_window(self, size: int) -> tuple[int, int]:
        if self.crop is None or self.crop >= size:
            return 0, 0
        top = int(self.rng.integers(0, size - self.crop + 1))
        left = int(self.rng.integers(0, size - self.crop + 1))
        return top, left

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        cp = self._pattern(row)

        image = cv2.imread(str(self.root / row["image_path"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(self.root / row["image_path"])
        size = image.shape[0]

        corners = np.fromstring(row["corners"], sep=" ").reshape(4, 2)
        # The back face is a different physical sheet surface; its FOLD file is
        # the front one, so mirror and swap before rasterising the labels.
        if row["face"] == "back":
            from origamicp.capture import back_face

            cp = back_face(cp)
        targets = build_targets(cp, corners, size)

        backlit = row["modality"] == BACKLIT
        # Withholding the light planes is the ablation that tests whether MV is
        # identifiable at all from one image: a mountain lit from the left is
        # the same picture as a valley lit from the right.
        light = light_channels(
            None if (backlit or not self.use_light) else float(row["light_azimuth_deg"]),
            size,
            None if (backlit or not self.use_light) else float(row["light_elevation_deg"]),
        )

        pixels = image.astype(np.float32) / 255.0
        if self.jitter:
            pixels = pixels * self.rng.uniform(0.9, 1.1) + self.rng.uniform(-0.05, 0.05)
            pixels = np.clip(pixels, 0.0, 1.0)

        stacked = np.concatenate([pixels[None], light], axis=0)

        top, left = self._crop_window(size)
        end = size if self.crop is None else min(top + self.crop, size)
        right = size if self.crop is None else min(left + self.crop, size)
        window = (slice(top, end), slice(left, right))

        return {
            "image": torch.from_numpy(np.ascontiguousarray(stacked[:, window[0], window[1]])),
            "crease": torch.from_numpy(targets["crease"][window].astype(np.float32)),
            "mv": torch.from_numpy(targets["mv"][window].astype(np.int64)),
            "sheet": torch.from_numpy(targets["sheet"][window].astype(np.float32)),
            "mv_valid": torch.tensor(0.0 if backlit else 1.0),
            "design_id": row["design_id"],
        }


def collate(batch: list[dict]) -> dict:
    """Default collation, but keeping ``design_id`` as a plain list of strings."""
    out = {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0]
        if key != "design_id"
    }
    out["design_id"] = [item["design_id"] for item in batch]
    return out
