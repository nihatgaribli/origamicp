"""Tests for the local flat-foldability conditions.

Each case isolates one condition: if a test fails we know which theorem the
checker got wrong, not just that "something is off".
"""

import json

import numpy as np
import pytest

from origamicp.core import CreasePattern
from origamicp.generate import single_vertex
from origamicp.verify import (
    big_little_big_violations,
    kawasaki_residual,
    maekawa_defect,
    validity_vs_tolerance,
    verify,
)

CENTER = 0  # single_vertex() always puts the interior vertex first


def test_star_geometry_round_trips():
    """The pattern builder must reproduce the sector angles it was given."""
    cp = single_vertex([60, 120, 120, 60], "MMMV")
    assert np.allclose(np.rad2deg(cp.sector_angles(CENTER)), [60, 120, 120, 60])
    assert list(cp.mv_around(CENTER)) == ["M", "M", "M", "V"]


def test_interior_and_boundary_classification():
    cp = single_vertex([90, 90, 90, 90], "MMMV")
    assert cp.interior_vertices() == [CENTER]
    assert cp.is_boundary_vertex(1)


def test_textbook_valid_vertex():
    """90/90/90/90 with three mountains and one valley is the canonical example."""
    report = verify(single_vertex([90, 90, 90, 90], "MMMV"))
    assert report.valid
    assert report.vertex_validity_rate == 1.0


def test_maekawa_catches_all_mountains():
    """Four mountains gives defect 4, so the vertex cannot fold flat."""
    report = verify(single_vertex([90, 90, 90, 90], "MMMM"))
    v = report.vertices[CENTER]
    assert v.maekawa_defect == 4
    assert not v.maekawa_ok
    assert v.kawasaki_ok  # geometry is fine; only the MV labels are wrong
    assert not report.valid


@pytest.mark.parametrize("mv", ["MVMV", "VMVM"])
def test_maekawa_catches_balanced_assignment(mv):
    """Equal mountains and valleys gives defect 0, which is also unfoldable."""
    report = verify(single_vertex([90, 90, 90, 90], mv))
    assert report.vertices[CENTER].maekawa_defect == 0
    assert not report.valid


def test_kawasaki_catches_bad_angles():
    """60/120/60/120 has alternating sum -120 degrees."""
    report = verify(single_vertex([60, 120, 60, 120], "MMMV"))
    v = report.vertices[CENTER]
    assert not v.kawasaki_ok
    assert np.isclose(np.rad2deg(v.kawasaki_residual), 120.0)
    assert v.maekawa_ok  # MV labels are fine; only the geometry is wrong


def test_kawasaki_accepts_unequal_but_alternating_angles():
    report = verify(single_vertex([60, 120, 120, 60], "MMMV"))
    assert report.vertices[CENTER].kawasaki_ok
    assert report.valid


def test_big_little_big_violation():
    """30/100/150/80 passes Maekawa and Kawasaki, so only BLB can reject it.

    Sector 0 (30 degrees) is strictly smaller than both neighbours, so the
    creases bounding it -- creases 0 and 1 -- must differ. Here both are M.
    """
    angles, mv = [30, 100, 150, 80], "MMMV"
    assert np.isclose(np.rad2deg(kawasaki_residual(np.deg2rad(angles))), 0.0)
    assert abs(maekawa_defect(np.array(list(mv)))) == 2

    report = verify(single_vertex(angles, mv))
    v = report.vertices[CENTER]
    assert v.kawasaki_ok and v.maekawa_ok
    assert v.blb_violations == [0]
    assert not report.valid


def test_big_little_big_satisfied_when_creases_differ():
    """Same geometry, but the little sector is now bounded by an M and a V."""
    report = verify(single_vertex([30, 100, 150, 80], "MVMM"))
    v = report.vertices[CENTER]
    assert v.blb_violations == []
    assert report.valid


def test_big_little_big_ignores_equal_neighbours():
    """A sector tied with its neighbour is not a *strict* minimum."""
    angles = np.deg2rad([90.0, 90.0, 90.0, 90.0])
    mv = np.array(list("MMMV"))
    assert big_little_big_violations(angles, mv) == []


def test_odd_degree_interior_vertex_is_rejected():
    report = verify(single_vertex([120, 120, 120], "MVM"))
    v = report.vertices[CENTER]
    assert not v.even_degree
    assert not v.kawasaki_ok
    assert not report.valid


def test_boundary_vertices_are_never_penalised():
    report = verify(single_vertex([90, 90, 90, 90], "MMMM"))
    assert all(r.ok for r in report.vertices if not r.interior)
    # The failure rate is computed over interior vertices only.
    assert report.vertex_validity_rate == 0.0
    assert len(report.interior) == 1


def test_degree_six_vertex():
    """Maekawa's +-2 holds at any degree, not just four."""
    angles = [80, 40, 70, 100, 20, 50]  # alternating sum: 80-40+70-100+20-50 = -20
    assert not verify(single_vertex(angles, "MMMMVV")).valid

    angles = [80, 40, 70, 100, 30, 40]  # 80-40+70-100+30-40 = 0
    report = verify(single_vertex(angles, "MMVMMV"))
    assert report.vertices[CENTER].kawasaki_ok
    assert report.vertices[CENTER].maekawa_ok


def test_tolerance_sweep_is_monotonic():
    """Loosening the tolerance can only ever admit more vertices."""
    cp = single_vertex([89.5, 90.5, 90.0, 90.0], "MMMV")  # residual = 1 degree
    sweep = validity_vs_tolerance(cp, np.array([0.1, 0.5, 2.0, 5.0]))
    rates = [rate for _, rate in sweep]
    assert rates == sorted(rates)
    assert rates[0] == 0.0 and rates[-1] == 1.0


def test_fold_round_trip(tmp_path):
    """FOLD is our interchange format with Flat-Folder, so it must survive I/O."""
    cp = single_vertex([30, 100, 150, 80], "MVMM")
    path = tmp_path / "vertex.fold"
    cp.to_fold(path)

    data = json.loads(path.read_text())
    assert data["frame_classes"] == ["creasePattern"]

    restored = CreasePattern.from_fold(path)
    assert np.allclose(restored.vertices, cp.vertices)
    assert np.array_equal(restored.edges, cp.edges)
    assert np.array_equal(restored.assignment, cp.assignment)
    assert verify(restored).valid


def test_rotation_invariance():
    """Validity is a property of the paper, not of how the photo was oriented."""
    cp = single_vertex([30, 100, 150, 80], "MVMM")
    angle = np.deg2rad(37.0)
    rot = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotated = CreasePattern(cp.vertices @ rot.T, cp.edges, cp.assignment)
    assert verify(rotated).valid


def test_malformed_patterns_are_rejected():
    with pytest.raises(ValueError, match="assignments"):
        CreasePattern([[0, 0], [1, 0]], [[0, 1]], ["M", "V"])
    with pytest.raises(ValueError, match="does not exist"):
        CreasePattern([[0, 0], [1, 0]], [[0, 5]], ["M"])
    with pytest.raises(ValueError, match="self-loop"):
        CreasePattern([[0, 0], [1, 0]], [[0, 0]], ["M"])
