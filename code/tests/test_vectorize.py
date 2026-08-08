"""Tests for line fitting, graph assembly and crease matching.

The vectoriser is checked against ground-truth masks throughout. That is the
ceiling: whatever it cannot recover from a perfect input, no amount of training
will recover from a predicted one, so this is where the assembly logic has to be
pinned down.
"""

import numpy as np
import pytest

from origamicp.core.cp import BOUNDARY, CreasePattern
from origamicp.generate.designs import PILOT_DESIGNS
from origamicp.models import build_targets, project_to_pixels
from origamicp.render import ScanStyle, render
from origamicp.vectorize import (
    detect_segments,
    extract_crease_pattern,
    refine_away_from_junctions,
    sheet_polygon,
)
from origamicp.vectorize.match import (
    crease_labels,
    crease_segments,
    match_creases,
)
from origamicp.verify import verify

SIZE = 512


def oracle_masks(name: str, rotation_deg: float = 0.0, seed: int = 1):
    """Ground-truth crease/MV/sheet masks for a pilot design."""
    cp = PILOT_DESIGNS[name]()
    _, corners = render(
        cp, ScanStyle(size_px=SIZE, rotation_deg=rotation_deg), np.random.default_rng(seed)
    )
    targets = build_targets(cp, corners, SIZE)
    crease_prob = targets["crease"].astype(np.float32)
    mv_label = np.where(targets["mv"] == 1, 0, 1).astype(np.int64)
    return cp, corners, crease_prob, mv_label, targets["sheet"]


def extract(name: str, rotation_deg: float = 0.0):
    cp, corners, crease_prob, mv_label, sheet = oracle_masks(name, rotation_deg)
    predicted, segments = extract_crease_pattern(crease_prob, mv_label, sheet)
    return cp, corners, predicted, segments


def interior(pattern):
    return verify(pattern, tol=np.deg2rad(3.0)).interior


# --------------------------------------------------------------------------
# line fitting
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["d01_pleat", "d02_vertex4", "d03_vertex6"])
def test_segment_count_matches_the_pattern(name):
    _, _, _, segments = extract(name)
    expected = sum(1 for kind in PILOT_DESIGNS[name]().assignment if kind != BOUNDARY)
    # Creases that cross are recovered as one line each, not one per piece.
    assert len(segments) == pytest.approx(expected, abs=2)


def test_dense_speckle_does_produce_spurious_creases():
    """Recorded as a limitation, because the hoped-for result did not hold.

    Line fitting was expected to discard the model's scattered false positives
    for free. Measured on sixty solid blobs it does not: enough of them fall
    into line to support seventeen spurious segments against three real ones.
    Confident detections at full probability are the worst case -- the model's
    actual false positives are fainter and more diffuse -- but the claim that
    vectorising removes speckle cannot be made on this evidence, and the real
    figure has to come from running the trained model end to end.

    What does survive is the true structure, which is what this pins down.
    """
    _, _, crease_prob, mv_label, sheet = oracle_masks("d02_vertex4")
    rng = np.random.default_rng(0)
    speckled = crease_prob.copy()
    for _ in range(60):
        y, x = rng.integers(60, SIZE - 60, size=2)
        if sheet[y, x]:
            speckled[y - 3 : y + 3, x - 3 : x + 3] = 1.0

    assert len(detect_segments(speckled, sheet)) > len(detect_segments(crease_prob, sheet))
    assert len(interior(extract_crease_pattern(speckled, mv_label, sheet)[0])) >= 1


def test_no_segments_on_a_blank_sheet():
    _, _, _, _, sheet = oracle_masks("d02_vertex4")
    assert detect_segments(np.zeros((SIZE, SIZE), np.float32), sheet) == []


def test_collinear_creases_split_at_a_gap():
    """Two creases on one line are two creases, not one long one."""
    sheet = np.zeros((256, 256), np.uint8)
    sheet[20:236, 20:236] = 1
    crease = np.zeros((256, 256), np.float32)
    crease[126:129, 30:100] = 1.0
    crease[126:129, 160:230] = 1.0

    segments = detect_segments(crease, sheet, min_length=30.0, max_gap=14.0)
    assert len(segments) == 2


# --------------------------------------------------------------------------
# graph assembly
# --------------------------------------------------------------------------


def test_sheet_polygon_follows_the_outline():
    _, _, _, _, sheet = oracle_masks("d02_vertex4")
    polygon = sheet_polygon(sheet)
    assert 4 <= len(polygon) <= 6  # square, plus the chamfered corner
    assert not (sheet > 0)[0].any()  # sanity: the sheet does not touch the frame


def test_degree_four_vertex_is_recovered_exactly():
    """The clean case: four creases meeting, geometry good enough for Kawasaki."""
    cp, _, predicted, _ = extract("d02_vertex4")
    recovered = interior(predicted)
    assert len(recovered) == 1
    assert recovered[0].degree == len(cp.incident_edges(cp.interior_vertices()[0]))
    assert verify(predicted, tol=np.deg2rad(3.0)).valid


@pytest.mark.parametrize("name", ["d03_vertex6", "d04_vertex8"])
def test_high_degree_vertices_recover_structurally(name):
    """Degree six and eight come back with the right shape but shaky angles.

    Many lines converging on one point is the vectoriser's weak case: each fit
    carries a fraction of a degree of error and the sector angles accumulate it,
    so Kawasaki's condition often misses at a three-degree tolerance even though
    the vertex has the right degree. Recorded as it is rather than skipped --
    this is the gap a junction-aware refit narrowed but did not close.
    """
    cp, _, predicted, _ = extract(name)
    recovered = interior(predicted)
    truth_degree = len(cp.incident_edges(cp.interior_vertices()[0]))

    assert 1 <= len(recovered) <= 2
    assert max(v.degree for v in recovered) == truth_degree


def test_pleat_yields_no_interior_vertices():
    """Parallel creases never cross, so there is nothing for them to make."""
    _, _, predicted, _ = extract("d01_pleat")
    assert verify(predicted).interior == []


def test_boundary_edges_close_the_outline():
    _, _, predicted, _ = extract("d03_vertex6")
    boundary = predicted.boundary_polygon()
    assert len(boundary) >= 4
    # A closed ring: every boundary vertex has exactly two boundary edges.
    for vertex in boundary:
        count = sum(
            1 for e in predicted.incident_edges(int(vertex)) if predicted.assignment[e] == BOUNDARY
        )
        assert count == 2


@pytest.mark.parametrize("rotation_deg", [0.0, 31.0, 67.0])
def test_crossing_patterns_are_stable_under_rotation(rotation_deg):
    """Miura is the case that works: every junction recovered, at any angle.

    Crossings behave far better than convergences. Two lines meeting near a
    right angle pin their intersection down tightly, whereas eight meeting at
    one point do not, which is why the corrugations pass here and the stars are
    tested only structurally.
    """
    cp, _, predicted, _ = extract("d05_miura_small", rotation_deg)
    recovered = interior(predicted)
    assert len(recovered) == len(cp.interior_vertices())
    # Most come back at the true degree of four; a few pick up an extra edge
    # from a neighbouring line that reaches slightly too far.
    assert sum(v.degree == 4 for v in recovered) >= 5
    assert all(3 <= v.degree <= 5 for v in recovered)


def test_mv_labels_survive_the_round_trip():
    """Labels come back off the map, so a wiring error here would invert them."""
    cp, corners, predicted, _ = extract("d03_vertex6")
    result = match_creases(
        crease_segments(cp, project_to_pixels(cp, corners)),
        crease_labels(cp),
        crease_segments(predicted),
        crease_labels(predicted),
    )
    assert result.recall > 0.8
    assert result.mv_accuracy > 0.9


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------


def segment(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y2]], dtype=np.float64)


def test_matching_rewards_an_exact_recovery():
    truth = [segment(0, 0, 100, 0), segment(0, 50, 100, 50)]
    result = match_creases(truth, ["M", "V"], list(truth), ["M", "V"])
    assert result.precision == 1.0 and result.recall == 1.0
    assert result.mv_accuracy == 1.0


def test_matching_notices_a_swapped_label():
    truth = [segment(0, 0, 100, 0)]
    result = match_creases(truth, ["M"], list(truth), ["V"])
    assert result.recall == 1.0
    assert result.mv_accuracy == 0.0


def test_matching_rejects_a_parallel_but_offset_crease():
    truth = [segment(0, 0, 100, 0)]
    result = match_creases(truth, ["M"], [segment(0, 40, 100, 40)], ["M"])
    assert result.matched == 0


def test_matching_rejects_a_crease_at_the_wrong_angle():
    truth = [segment(0, 0, 100, 0)]
    result = match_creases(truth, ["M"], [segment(0, 0, 70, 70)], ["M"])
    assert result.matched == 0


def test_matching_needs_enough_overlap():
    truth = [segment(0, 0, 100, 0)]
    assert match_creases(truth, ["M"], [segment(0, 0, 20, 0)], ["M"]).matched == 0
    assert match_creases(truth, ["M"], [segment(0, 0, 80, 0)], ["M"]).matched == 1


def test_duplicate_predictions_cost_precision():
    """One prediction may be spent on one truth, so copies are false positives."""
    truth = [segment(0, 0, 100, 0)]
    doubled = [segment(0, 0, 100, 0), segment(0, 1, 100, 1)]
    result = match_creases(truth, ["M"], doubled, ["M", "M"])
    assert result.recall == 1.0
    assert result.precision == 0.5


def test_density_filter_rejects_lines_through_scattered_noise():
    """A crease is a solid band; a line through blobs is mostly empty.

    Density -- supporting pixels per unit length -- separates them without
    reference to the image, which is why it survives where a blob-size filter
    would not.
    """
    sheet = np.zeros((300, 300), np.uint8)
    sheet[20:280, 20:280] = 1

    solid = np.zeros((300, 300), np.float32)
    solid[148:151, 40:260] = 1.0

    # Specks close enough that the gap rule lets them form one line; only their
    # density gives them away.
    dotted = solid.copy()
    for x in range(40, 260, 16):
        dotted[78:81, x : x + 6] = 1.0

    dense = detect_segments(dotted, sheet, min_length=30.0, min_density=0.0)
    filtered = detect_segments(dotted, sheet, min_length=30.0, min_density=2.0)

    assert len(dense) == 2, "the speck trail should form a line without the filter"
    assert len(filtered) == 1, "and be rejected with it"
    assert filtered[0].length > 180  # the real crease survives
    assert filtered[0].density > 2.5


def test_density_is_carried_through_refinement():
    """Refitting rebuilds the segments, and once dropped the field read zero."""
    _, _, crease_prob, _, sheet = oracle_masks("d05_miura_small")
    segments = detect_segments(crease_prob, sheet)
    refined = refine_away_from_junctions(segments, crease_prob, sheet)
    assert all(s.density > 1.0 for s in refined)
