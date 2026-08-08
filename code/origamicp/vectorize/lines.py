"""Turn a predicted crease probability map into straight line segments.

Origami creases are straight by construction -- paper folds along lines -- so
the vectoriser fits lines rather than tracing curves. That assumption is what
removes the model's characteristic error: the spurious blobs it sprinkles over
paper undulation are not collinear with anything, so no line claims them and
they disappear without a hand-tuned blob filter.

Only OpenCV and NumPy are used. Thinning lives in opencv-contrib and
skeletonisation in scikit-image, neither of which the project needs otherwise;
refitting each cluster by weighted PCA over its own pixels recovers the line
centre just as well, because a symmetric ridge has its centroid on the ridge.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Segment:
    start: np.ndarray
    end: np.ndarray
    support: float  # mean predicted crease probability along the segment
    density: float = 0.0  # supporting pixels per unit length

    # Density is what separates a crease from a line drawn through scattered
    # noise. A real crease is a solid band a few pixels wide, so it contributes
    # roughly its thickness in pixels for every pixel of length. A line that
    # happens to pass through a handful of unrelated blobs contributes far
    # less, however convincing its Hough score looked.

    @property
    def direction(self) -> np.ndarray:
        delta = self.end - self.start
        return delta / max(float(np.linalg.norm(delta)), 1e-9)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))


def _canonical(point: np.ndarray, direction: np.ndarray) -> tuple[float, float]:
    """Represent a line as ``(theta, rho)`` with theta in [0, pi)."""
    normal = np.array([-direction[1], direction[0]])
    theta = float(np.arctan2(normal[1], normal[0]))
    rho = float(normal @ point)
    if theta < 0:
        theta += np.pi
        rho = -rho
    return theta, rho


def _line_gap(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """Angle and offset difference, handling the wrap at theta = pi.

    A line at theta just under pi and one at theta just over zero are nearly
    parallel even though their thetas differ by almost pi -- and their rhos
    then carry opposite signs.
    """
    delta = abs(a[0] - b[0])
    if delta <= np.pi / 2:
        return delta, abs(a[1] - b[1])
    return np.pi - delta, abs(a[1] + b[1])


def _cluster(
    lines: list[tuple[tuple[float, float], float]], theta_tol: float, rho_tol: float
) -> list[tuple[float, float]]:
    """Distinct lines, keeping the longest of each near-identical group.

    Hough returns many fragments per crease; taking the longest as the
    representative means the refit starts from the best-supported estimate.
    """
    representatives: list[tuple[float, float]] = []
    for line, _ in sorted(lines, key=lambda item: -item[1]):
        if not any(
            d_theta <= theta_tol and d_rho <= rho_tol
            for d_theta, d_rho in (_line_gap(line, rep) for rep in representatives)
        ):
            representatives.append(line)
    return representatives


def _refit(
    points: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Weighted total-least-squares line: returns ``(centroid, direction)``."""
    if len(points) < 8:
        return None
    total = float(weights.sum())
    if total <= 1e-9:
        return None
    centroid = (points * weights[:, None]).sum(axis=0) / total
    centred = points - centroid
    covariance = (centred * weights[:, None]).T @ centred / total
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    return centroid, direction / max(float(np.linalg.norm(direction)), 1e-9)


def detect_segments(
    crease_prob: np.ndarray,
    sheet_mask: np.ndarray,
    threshold: float = 0.5,
    min_length: float = 18.0,
    band: float = 3.0,
    max_gap: float = 14.0,
    min_density: float = 0.0,
    theta_tol: float = np.deg2rad(2.5),
    rho_tol: float = 6.0,
    seed_theta_tol: float = np.deg2rad(7.0),
    seed_rho_tol: float = 12.0,
) -> list[Segment]:
    """Fit straight segments to the predicted creases.

    ``band`` is how far from a candidate line a pixel may sit and still be
    counted as its evidence; ``max_gap`` is how large a break along a line may
    be before it is treated as two separate creases rather than one.
    """
    binary = ((crease_prob > threshold) & (sheet_mask > 0)).astype(np.uint8)
    if not binary.any():
        return []

    hough = cv2.HoughLinesP(
        binary * 255,
        rho=1,
        theta=np.pi / 360,
        threshold=max(20, int(min_length)),
        minLineLength=min_length,
        maxLineGap=max_gap,
    )
    if hough is None:
        return []

    candidates = []
    for x1, y1, x2, y2 in hough.reshape(-1, 4).astype(np.float64):
        start, end = np.array([x1, y1]), np.array([x2, y2])
        length = float(np.linalg.norm(end - start))
        if length < 1e-6:
            continue
        candidates.append((_canonical(start, (end - start) / length), length))

    representatives = _cluster(candidates, seed_theta_tol, seed_rho_tol)

    ys, xs = np.nonzero(binary)
    pixels = np.stack([xs, ys], axis=1).astype(np.float64)
    weights_all = crease_prob[ys, xs].astype(np.float64)

    # Refit each seed, then deduplicate again on the refined parameters.
    # Hough returns several fragments per crease whose angles differ by more
    # than a tight tolerance allows, so clustering the raw output either splits
    # one crease into several lines or, if loosened enough to stop that, merges
    # genuinely different ones. Refitting first collapses the fragments onto a
    # common line, and the second pass can then be strict.
    refined: list[tuple[np.ndarray, np.ndarray]] = []
    for theta, rho in representatives:
        normal = np.array([np.cos(theta), np.sin(theta)])
        centroid = normal * rho
        direction = np.array([-normal[1], normal[0]])

        for _ in range(2):
            offsets = np.abs((pixels - centroid) @ np.array([-direction[1], direction[0]]))
            selected = offsets <= band
            if selected.sum() < 8:
                break
            fitted = _refit(pixels[selected], weights_all[selected])
            if fitted is None:
                break
            centroid, direction = fitted
        else:
            canonical = _canonical(centroid, direction)
            if not any(
                d_theta <= theta_tol and d_rho <= rho_tol
                for d_theta, d_rho in (
                    _line_gap(canonical, _canonical(c, d)) for c, d in refined
                )
            ):
                refined.append((centroid, direction))

    segments: list[Segment] = []
    for centroid, direction in refined:
        # Pixels are shared between lines rather than claimed by the first:
        # two creases that cross overlap at the junction, and taking those
        # pixels from the second line opens a gap there, which the gap rule
        # would then read as two separate creases.
        offsets = np.abs((pixels - centroid) @ np.array([-direction[1], direction[0]]))
        selected = offsets <= band
        if selected.sum() < 8:
            continue

        chosen = pixels[selected]
        weights = weights_all[selected]
        t = (chosen - centroid) @ direction
        order = np.argsort(t)
        t, weights = t[order], weights[order]

        breaks = np.nonzero(np.diff(t) > max_gap)[0]
        for lo, hi in zip(
            np.concatenate([[0], breaks + 1]), np.concatenate([breaks + 1, [len(t)]])
        ):
            if hi - lo < 4:
                continue
            span = float(t[hi - 1] - t[lo])
            if span < min_length:
                continue
            density = (hi - lo) / max(span, 1e-9)
            if density < min_density:
                continue
            segments.append(
                Segment(
                    start=centroid + t[lo] * direction,
                    end=centroid + t[hi - 1] * direction,
                    support=float(weights[lo:hi].mean()),
                    density=density,
                )
            )

    segments.sort(key=lambda s: -s.length)
    return _deduplicate(segments, theta_tol, band * 2.0)


def refine_away_from_junctions(
    segments: list[Segment],
    crease_prob: np.ndarray,
    sheet_mask: np.ndarray,
    threshold: float = 0.5,
    band: float = 3.0,
    junction_radius: float = 14.0,
    min_angle: float = np.deg2rad(12.0),
) -> list[Segment]:
    """Re-fit each line using only pixels well away from any crossing.

    Where creases meet, their pixels overlap, and a fit that includes them is
    pulled toward its neighbours. At a degree-eight vertex eight lines drag on
    each other and the recovered sector angles come out several degrees wrong --
    enough to fail Kawasaki's condition on a pattern that is structurally
    correct. Refitting on the clean stretches removes that bias; the extent is
    kept from the first pass, since the junction is exactly where the crease
    genuinely reaches.
    """
    if len(segments) < 2:
        return segments

    junctions = []
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            a, b = segments[i], segments[j]
            da, db = a.direction, b.direction
            if abs(float(da @ db)) > np.cos(min_angle):
                continue
            denominator = da[0] * db[1] - da[1] * db[0]
            if abs(denominator) < 1e-9:
                continue
            delta = b.start - a.start
            t = (delta[0] * db[1] - delta[1] * db[0]) / denominator
            if -junction_radius <= t <= a.length + junction_radius:
                junctions.append(a.start + t * da)
    if not junctions:
        return segments

    binary = (crease_prob > threshold) & (sheet_mask > 0)
    ys, xs = np.nonzero(binary)
    pixels = np.stack([xs, ys], axis=1).astype(np.float64)
    weights = crease_prob[ys, xs].astype(np.float64)

    junction_array = np.array(junctions)
    # Chunked over pixels rather than allocating pixels x junctions at once. On
    # a clean mask that product is small; on a noisy one -- the learning-free
    # baseline at a permissive threshold -- it is thousands of junctions against
    # hundreds of thousands of pixels, and the whole run dies on the allocation
    # rather than returning a poor result the caller could report.
    clean = np.empty(len(pixels), dtype=bool)
    budget = max(1, 4_000_000 // max(len(junction_array), 1))
    for lo in range(0, len(pixels), budget):
        block = pixels[lo : lo + budget]
        distances = np.linalg.norm(block[:, None, :] - junction_array[None, :, :], axis=2)
        clean[lo : lo + budget] = distances.min(axis=1) > junction_radius

    refined = []
    for segment in segments:
        normal = np.array([-segment.direction[1], segment.direction[0]])
        near = (np.abs((pixels - segment.start) @ normal) <= band) & clean
        fitted = _refit(pixels[near], weights[near]) if near.sum() >= 8 else None
        if fitted is None:
            refined.append(segment)
            continue
        centroid, direction = fitted
        if float(direction @ segment.direction) < 0:
            direction = -direction
        # Keep the original extent, expressed on the corrected line.
        t0 = float((segment.start - centroid) @ direction)
        t1 = float((segment.end - centroid) @ direction)
        refined.append(
            Segment(
                centroid + t0 * direction,
                centroid + t1 * direction,
                segment.support,
                segment.density,
            )
        )
    return refined


def _deduplicate(
    segments: list[Segment], angle_tol: float, offset_tol: float, min_overlap: float = 0.6
) -> list[Segment]:
    """Drop segments that duplicate a longer one.

    Deduplication happens here, on segments, rather than earlier in (theta, rho)
    space. Rho is measured from the image origin, so for a line far from it a
    fraction of a degree of angle error moves rho by several pixels -- a
    tolerance loose enough to merge two fits of the same crease is then loose
    enough to merge two different creases. Comparing the segments directly
    asks the question that actually matters: do these two occupy the same
    stretch of the same line?

    Collinear creases separated by a gap do not overlap, so they survive.
    """
    kept: list[Segment] = []
    for segment in segments:  # already longest-first
        direction = segment.direction
        duplicate = False
        for other in kept:
            if abs(float(direction @ other.direction)) < np.cos(angle_tol):
                continue
            normal = np.array([-other.direction[1], other.direction[0]])
            if max(abs((np.stack([segment.start, segment.end]) - other.start) @ normal)) > offset_tol:
                continue
            t = np.sort((np.stack([segment.start, segment.end]) - other.start) @ other.direction)
            overlap = max(0.0, min(t[1], other.length) - max(t[0], 0.0))
            if overlap >= min_overlap * segment.length:
                duplicate = True
                break
        if not duplicate:
            kept.append(segment)
    return kept
