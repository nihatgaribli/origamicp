"""Tests for the pilot design set.

A template that cannot fold flat wastes a sheet, an hour of folding and a scan
slot, and the mistake only surfaces weeks later. So validity is a test, not a
thing we check by eye.
"""

import numpy as np
import pytest

from origamicp.capture import back_face
from origamicp.core import BOUNDARY
from origamicp.generate.designs import (
    PILOT_DESIGNS,
    SHEET_MM,
    miura,
    pleat,
    single_vertex_sheet,
)
from origamicp.render import to_svg
from origamicp.verify import verify


@pytest.mark.parametrize("name", sorted(PILOT_DESIGNS))
def test_every_pilot_design_folds_flat(name):
    report = verify(PILOT_DESIGNS[name]())
    assert report.valid, f"{name}: {report.summary()}"


@pytest.mark.parametrize("name", sorted(PILOT_DESIGNS))
def test_every_pilot_design_fits_the_sheet(name):
    cp = PILOT_DESIGNS[name]()
    assert cp.vertices.min() >= -1e-6
    assert cp.vertices.max() <= SHEET_MM + 1e-6


@pytest.mark.parametrize("name", sorted(PILOT_DESIGNS))
def test_printed_template_is_also_valid(name):
    """The template is the back face; folding it must be a well-posed task."""
    assert verify(back_face(PILOT_DESIGNS[name]())).valid


def test_single_vertex_sheet_is_off_centre():
    """Centred patterns register ambiguously, so the ladder must avoid them."""
    cp = PILOT_DESIGNS["d02_vertex4"]()
    interior = cp.interior_vertices()
    assert len(interior) == 1
    assert not np.allclose(cp.vertices[interior[0]], [SHEET_MM / 2, SHEET_MM / 2])


def test_single_vertex_creases_reach_the_boundary():
    cp = single_vertex_sheet([60, 120, 120, 60], "MMMV")
    centre = cp.interior_vertices()[0]
    for edge in cp.incident_edges(centre):
        endpoint = cp.vertices[cp.other_end(edge, centre)]
        on_edge = np.isclose(endpoint, 0.0).any() or np.isclose(endpoint, SHEET_MM).any()
        assert on_edge, f"crease stops inside the sheet at {endpoint}"


def test_single_vertex_sheet_rejects_bad_input():
    with pytest.raises(ValueError, match="sum to"):
        single_vertex_sheet([90, 90, 90], "MMV")
    with pytest.raises(ValueError, match="MV labels"):
        single_vertex_sheet([90, 90, 90, 90], "MMV")


def test_pleat_has_no_interior_vertices():
    """Rung zero isolates crease detection from any MV reasoning."""
    report = verify(pleat(8))
    assert report.interior == []
    assert report.valid
    assert np.isnan(report.vertex_validity_rate)


def test_pleat_alternates_mountains_and_valleys():
    cp = pleat(6)
    folds = [a for a in cp.assignment if a != BOUNDARY]
    assert folds == list("MVMVMV")


@pytest.mark.parametrize("n", [2, 4, 6, 8])
def test_miura_is_valid_at_every_size(n):
    report = verify(miura(n, n))
    assert report.valid
    assert len(report.interior) == (n - 1) ** 2


def test_miura_vertices_are_three_to_one():
    """Maekawa's split is what the alternating-by-row rule is chosen to give."""
    cp = miura(4, 4)
    report = verify(cp)
    for vertex in report.interior:
        assert vertex.degree == 4
        assert abs(vertex.maekawa_defect) == 2


def test_miura_sheet_is_square():
    """Flattening the top and bottom rows keeps the outline a true square."""
    cp = miura(4, 4)
    lo, hi = cp.vertices.min(axis=0), cp.vertices.max(axis=0)
    assert np.allclose(lo, [0.0, 0.0])
    assert np.allclose(hi, [SHEET_MM, SHEET_MM])


def test_svg_is_written_at_physical_size(tmp_path):
    cp = PILOT_DESIGNS["d05_miura_small"]()
    path = to_svg(cp, tmp_path / "d05.svg", label="d05", sheet_mm=SHEET_MM)
    svg = path.read_text(encoding="utf-8")

    assert 'width="170.0mm"' in svg  # 150mm sheet plus 10mm margins
    assert svg.count("<line") == cp.n_edges
    assert "stroke-dasharray" in svg
    assert ">d05<" in svg
