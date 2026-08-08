"""A small U-Net with two heads: crease detection and mountain/valley.

Two heads rather than one three-class output. Detecting a crease and calling it
a mountain are different problems -- one is edge detection, the other reads a
shading asymmetry a few grey levels wide -- and separating them lets the MV loss
be supervised only where a crease actually is. A single three-class head would
spend almost all of its gradient on background pixels and report one number that
hides which half of the task is failing.

Deliberately small. The dataset is synthetic and unlimited, the real set will be
a few hundred sheets, and an 8 GB laptop GPU has to fine-tune it; capacity is
not the binding constraint here, and a compact model leaves room for the full
768-pixel sheet at evaluation time.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class CreaseUNet(nn.Module):
    """Input: greyscale scan plus two light-direction planes. Output: two maps."""

    def __init__(self, in_channels: int = 3, width: int = 32, depth: int = 4) -> None:
        super().__init__()
        widths = [width * 2**i for i in range(depth + 1)]

        self.encoders = nn.ModuleList()
        channels = in_channels
        for w in widths[:-1]:
            self.encoders.append(_block(channels, w))
            channels = w
        self.bottleneck = _block(channels, widths[-1])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        channels = widths[-1]
        for w in reversed(widths[:-1]):
            self.ups.append(nn.ConvTranspose2d(channels, w, 2, stride=2))
            self.decoders.append(_block(w * 2, w))
            channels = w

        self.crease_head = nn.Conv2d(channels, 1, 1)
        self.mv_head = nn.Conv2d(channels, 2, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = decoder(torch.cat([x, skip], dim=1))

        return {"crease": self.crease_head(x).squeeze(1), "mv": self.mv_head(x)}


def losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    crease_pos_weight: float = 12.0,
    mv_weight: float = 1.0,
    flip_invariant_mv: bool = False,
) -> dict[str, torch.Tensor]:
    """Crease BCE over the sheet, MV cross-entropy on crease pixels only.

    Both losses are restricted to the paper: the dark backing is trivially
    background and would otherwise supply most of the gradient. ``pos_weight``
    offsets the remaining imbalance -- creases still cover only a few percent of
    a sheet.

    The MV term is masked twice, by the ground-truth crease mask and by
    ``mv_valid``. The second mask drops back-lit sheets, whose MV labels are not
    recoverable from the image by anyone.

    ``flip_invariant_mv`` scores each sheet against the better of the two global
    assignments instead of against the labelled one. It exists for the case the
    light-direction ablation creates, where the sign of every crease is fixed
    only up to inverting all of them at once. Per-pixel cross-entropy has its
    minimum at one half on every crease there -- refusing to decide is optimal,
    and the argmax that follows is then settled by whatever asymmetry the
    features happen to carry. Taking the minimum over the two global labellings
    removes the part of the target that the image cannot determine, and leaves
    the part it can: which creases agree with which. Its floor is not ln 2.

    A training crop is not a whole sheet, but every crease within one shares the
    sheet's global sign, so the minimum is the same quantity restricted to the
    crop.
    """
    sheet = batch["sheet"]
    crease = batch["crease"]

    per_pixel = F.binary_cross_entropy_with_logits(
        outputs["crease"],
        crease,
        pos_weight=torch.tensor(crease_pos_weight, device=crease.device),
        reduction="none",
    )
    crease_loss = (per_pixel * sheet).sum() / sheet.sum().clamp(min=1.0)

    # mv targets are 1/2; shift to 0/1 for a two-way cross-entropy.
    mv_target = (batch["mv"].clamp(min=1) - 1).long()
    mv_per_pixel = F.cross_entropy(outputs["mv"], mv_target, reduction="none")
    mv_mask = crease * sheet * batch["mv_valid"][:, None, None]

    if flip_invariant_mv:
        flipped = F.cross_entropy(outputs["mv"], 1 - mv_target, reduction="none")
        # Per sample, not per batch: the two sheets in a batch are folded
        # independently and each gets its own sign.
        covered = mv_mask.sum(dim=(1, 2))
        weight = covered.clamp(min=1.0)
        as_labelled = (mv_per_pixel * mv_mask).sum(dim=(1, 2)) / weight
        as_inverted = (flipped * mv_mask).sum(dim=(1, 2)) / weight
        scored = (covered > 0).float()
        mv_loss = (torch.minimum(as_labelled, as_inverted) * scored).sum() / scored.sum().clamp(
            min=1.0
        )
    else:
        mv_loss = (mv_per_pixel * mv_mask).sum() / mv_mask.sum().clamp(min=1.0)

    return {
        "crease": crease_loss,
        "mv": mv_loss,
        "total": crease_loss + mv_weight * mv_loss,
    }
