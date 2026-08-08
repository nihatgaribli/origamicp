#!/usr/bin/env python
"""The renderer's difficulty statistic, measured on photographs of real creases.

    python scripts/real_crease_snr.py --images ../data/real/warpdoc/WarpDoc/image/fold

``measure_crease_snr`` answers whether the mountain/valley cue can be seen at
all: the brightness step across a crease, divided by the spread of the same
measurement taken on blank paper. It is the handle the renderer is supposed to
be calibrated by, and it has only ever been evaluated on the renderer's own
output -- which makes it a self-consistency check rather than a calibration.

This measures the same quantity on real photographs, so the synthetic range can
be compared against something. It is deliberately not a transfer study. These
images have no ground-truth crease graph, their lighting is uncontrolled and its
direction unknown, and the sheets are printed rather than blank. What survives
all of that is the magnitude of the contrast, which is the one number the
renderer's ``crease_height``, ``undulation`` and ``fiber_noise`` are set by.

Three things have to hold for the comparison to mean anything, and each is
enforced rather than assumed:

**The offset must be the same distance on the paper.** The synthetic sheets put
about 3.5 px in a millimetre; a 3024-pixel photograph of a page puts eight or
nine there. Reusing 3.5 px would sample inside the crease itself and measure the
fold rather than the step across it, so the offset is derived from the detected
page width.

**Both sides of a sample must be blank paper.** Text is a far stronger local
signal than a crease, and the noise term is meant to be the paper's own texture.
Samples are taken only where the surrounding window is free of print.

**The background must be measured exactly as the signal is.** Fifteen samples
averaged along a line, as in the synthetic version -- comparing that against
single-pixel noise would understate the ratio by roughly sqrt(15).

Creases are found rather than known, which is the weak point: a false line
through clutter would be measured as though it were a fold. ``--debug`` writes
overlays so the detections can be looked at instead of trusted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# A4 short side. The datasets mix page sizes, so this sets the pixels-per-mm
# scale only approximately -- close enough for an offset that just has to land
# outside the crease and inside the paper.
SHEET_SHORT_MM = 210.0
OFFSET_MM = 1.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-side", type=int, default=1600, help="downscale before work")
    parser.add_argument("--debug", type=Path, help="write detection overlays here")
    return parser.parse_args(argv)


def page_mask(image: np.ndarray) -> tuple[np.ndarray, float] | None:
    """The sheet as a filled mask, and its short side in pixels.

    Segmented on saturation rather than brightness. A sheet on a wooden desk is
    not reliably brighter than the desk -- under warm indoor light the wood can
    match it -- but it is far less coloured, and that separates the two cleanly
    where a grey-level threshold leaked across onto the table and had the
    paper's texture measured on wood grain.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    binary = (
        (cv2.GaussianBlur(saturation, (0, 0), 3) < 60)
        & (cv2.GaussianBlur(value, (0, 0), 3) > 90)
    ).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 0.08 * image.shape[0] * image.shape[1]:
        return None

    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    return mask, float(min(w, h))


def blank_mask(gray: np.ndarray, paper: np.ndarray, window: int) -> np.ndarray:
    """Paper that carries no print, eroded clear of the sheet's own edges.

    The local spread of a printed page is sharply bimodal -- blank paper sits
    near half a grey level, print above twenty -- so Otsu finds the valley
    between them without a threshold having to be guessed per image.

    A percentile cutoff was tried first and was wrong in a way worth recording:
    at the median it discards half the sheet, and the half it discards is the
    half with structure in it. Creases have structure. The mask was removing
    exactly what the measurement was looking for.
    """
    window = max(3, window | 1)
    mean = cv2.blur(gray, (window, window))
    variance = cv2.blur(gray * gray, (window, window)) - mean * mean
    local_std = np.sqrt(np.maximum(variance, 0))

    inside = cv2.erode(paper, np.ones((window * 2 + 1, window * 2 + 1), np.uint8))
    candidate = local_std[inside > 0]
    if candidate.size < 1000:
        return np.zeros_like(paper)

    scale = 255.0 / max(float(candidate.max()), 1e-6)
    cutoff, _ = cv2.threshold(
        np.clip(candidate * scale, 0, 255).astype(np.uint8),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    cutoff /= scale

    blank = ((local_std <= cutoff) & (inside > 0)).astype(np.uint8) * 255
    # Stand back from the print as well as from the page edge: a sample taken a
    # millimetre from a letter is measuring the letter.
    return cv2.erode(blank, np.ones((window, window), np.uint8))


def fold_lines(gray: np.ndarray, blank: np.ndarray, min_length: float) -> list[np.ndarray]:
    """Long straight brightness steps lying in blank paper.

    Restricted to blank paper, which does most of the work: searched over the
    whole sheet, Hough locks onto the rows and columns of the text block and
    returns hundreds of lines that are typography, not folds. Inside the margins
    there is nothing straight except the folds and the page edge.
    """
    smooth = cv2.GaussianBlur(gray, (0, 0), 2)
    edges = cv2.Canny(smooth.astype(np.uint8), 10, 40)
    edges[blank == 0] = 0

    hough = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,
        threshold=int(min_length),
        minLineLength=min_length,
        maxLineGap=min_length / 3,
    )
    if hough is None:
        return []
    return [line.astype(np.float64).reshape(2, 2) for line in hough.reshape(-1, 4)]


def _step(gray: np.ndarray, blank: np.ndarray, point, normal, offset: float):
    """Brightness difference across ``point``, or None if either side is unusable."""
    height, width = gray.shape
    values = []
    for sign in (1.0, -1.0):
        p = point + normal * offset * sign
        x, y = int(round(p[0])), int(round(p[1]))
        if not (1 <= x < width - 1 and 1 <= y < height - 1) or blank[y, x] == 0:
            return None
        values.append(float(gray[y, x]))
    return values[0] - values[1]


def _line_signal(gray, blank, segment, offset):
    start, end = segment
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 8 * offset:
        return None
    normal = np.array([-direction[1], direction[0]]) / length

    steps = [
        s
        for t in np.linspace(0.2, 0.8, 15)
        if (s := _step(gray, blank, start + direction * t, normal, offset)) is not None
    ]
    # Partial coverage means the line ran out of blank paper; a mean over three
    # samples is not the same statistic as a mean over fifteen.
    return abs(float(np.mean(steps))) if len(steps) >= 12 else None


def _background(gray, blank, offset, rng, attempts=800):
    ys, xs = np.nonzero(blank)
    if len(xs) < 100:
        return []
    span = 15 * offset
    values = []
    for _ in range(attempts):
        index = rng.integers(len(xs))
        point = np.array([float(xs[index]), float(ys[index])])
        phi = rng.uniform(0, 2 * np.pi)
        along = np.array([np.cos(phi), np.sin(phi)])
        normal = np.array([-along[1], along[0]])
        steps = [
            s
            for t in np.linspace(-0.5, 0.5, 15)
            if (s := _step(gray, blank, point + along * span * t, normal, offset)) is not None
        ]
        if len(steps) >= 12:
            values.append(float(np.mean(steps)))
    return values


def measure(path: Path, max_side: int, rng, debug: Path | None):
    image = cv2.imread(str(path))
    if image is None:
        return None
    scale = max_side / max(image.shape[:2])
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

    found = page_mask(image)
    if found is None:
        return None
    paper, short_side = found

    per_mm = short_side / SHEET_SHORT_MM
    offset = max(2.0, OFFSET_MM * per_mm)
    blank = blank_mask(gray, paper, window=int(max(3, 2 * per_mm)))
    segments = fold_lines(gray, blank, min_length=max(20.0, 20.0 * per_mm))
    if not segments:
        return None

    signals = [s for seg in segments if (s := _line_signal(gray, blank, seg, offset)) is not None]
    if not signals:
        return None
    background = _background(gray, blank, offset, rng)
    if len(background) < 50:
        return None

    signal = float(np.mean(signals))
    noise = float(np.std(background))
    if noise <= 1e-9:
        return None

    if debug is not None:
        overlay = image.copy()
        overlay[blank > 0] = (0.7 * overlay[blank > 0] + 0.3 * np.array([0, 255, 0])).astype(
            np.uint8
        )
        for seg in segments:
            cv2.line(
                overlay,
                tuple(seg[0].astype(int)),
                tuple(seg[1].astype(int)),
                (0, 0, 255),
                2,
            )
        debug.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug / f"{path.stem}.jpg"), overlay)

    return signal, noise, len(signals)


def main(argv=None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(0)

    paths = sorted(
        p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"no images under {args.images}")
        return 1

    rows = []
    for path in paths:
        result = measure(path, args.max_side, rng, args.debug)
        if result is not None:
            signal, noise, count = result
            rows.append((path.name, signal, noise, signal / noise, count))

    print(f"\ncrease signal-to-texture on {len(rows)} of {len(paths)} photographs\n")
    print(f"  {'image':<16}{'signal':>9}{'noise':>9}{'ratio':>9}{'lines':>7}")
    for name, signal, noise, ratio, count in rows[:20]:
        print(f"  {name:<16}{signal:>9.2f}{noise:>9.2f}{ratio:>9.2f}{count:>7}")
    if len(rows) > 20:
        print(f"  ... {len(rows) - 20} more")

    if rows:
        ratios = np.array([r[3] for r in rows])
        print(
            f"\n  ratio  median {np.median(ratios):.2f}  "
            f"mean {ratios.mean():.2f}  "
            f"10-90% {np.percentile(ratios, 10):.2f} - {np.percentile(ratios, 90):.2f}  "
            f"range {ratios.min():.2f} - {ratios.max():.2f}"
        )
        print("\n  synthetic set, for comparison:  0.46 - 7.42 (results/difficulty.txt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
