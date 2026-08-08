"""Segmentation metrics reported separately for detection and for MV.

Averaging the two into one number would hide the finding the project exists to
measure: crease detection is expected to be easy and mountain/valley hard, and
the gap between them is the result, not a nuisance.

MV accuracy is scored only where the model and the ground truth agree a crease
is present. Scoring it everywhere would let a model that detects nothing look
perfect, and scoring it on missed creases would double-count detection errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Counts:
    """Running totals, so metrics come from whole epochs rather than batch means."""

    true_positive: float = 0.0
    false_positive: float = 0.0
    false_negative: float = 0.0
    mv_correct: float = 0.0
    mv_total: float = 0.0
    _samples: int = field(default=0)

    def update(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        threshold: float = 0.5,
    ) -> None:
        sheet = batch["sheet"].bool()
        truth = batch["crease"].bool() & sheet
        predicted = (outputs["crease"].sigmoid() > threshold) & sheet

        self.true_positive += float((predicted & truth).sum())
        self.false_positive += float((predicted & ~truth).sum())
        self.false_negative += float((~predicted & truth).sum())

        valid = batch["mv_valid"].bool()[:, None, None]
        scorable = predicted & truth & valid
        if scorable.any():
            predicted_mv = outputs["mv"].argmax(dim=1)
            target_mv = batch["mv"].clamp(min=1) - 1
            self.mv_correct += float(((predicted_mv == target_mv) & scorable).sum())
            self.mv_total += float(scorable.sum())

        self._samples += len(batch["crease"])

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def mv_accuracy(self) -> float:
        """Chance is 0.5. A model reading texture rather than shading lands there."""
        return self.mv_correct / self.mv_total if self.mv_total else float("nan")

    def summary(self) -> str:
        return (
            f"crease P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f} | "
            f"MV acc={self.mv_accuracy:.3f} over {int(self.mv_total):,} px | "
            f"{self._samples} images"
        )
