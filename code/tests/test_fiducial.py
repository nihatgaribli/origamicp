"""Tests for the corner chamfer that pins a sheet's orientation.

The failure this prevents is the quiet kind. A symmetric pattern registered at
the wrong rotation still lands on every crease, so the alignment score looks
fine while every mountain has been labelled a valley. Nothing further down the
pipeline checks the orientation, so it has to be settled here.
"""

import numpy as np
import pytest

from origamicp.capture import back_face, detect_sheet_outline, register
from origamicp.core import BOUNDARY, CreasePattern
from origamicp.generate import chamfer, chamfered_corner, is_chamfered
from origamicp.generate.designs import PILOT_DESIGNS, SHEET_MM, miura, pleat
from origamicp.render import ScanStyle, render
from origamicp.render.scan import _project
from origamicp.verify import verify


def plain_square() -> CreasePattern:
    vertices = np.array([[0, 0], [150, 0], [150, 150], [0, 150]], dtype=np.float64)
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    return CreasePattern(vertices, edges, np.array(list("BBBB")))


def test_chamfer_replaces_the_corner_with_a_cut():
    cut = chamfer(plain_square(), 8.0)
    assert cut.n_vertices == 5  # one corner out, two cut ends in
    assert cut.n_edges == 5
    assert all(a == BOUNDARY for a in cut.assignment)
    assert len(cut.boundary_polygon()) == 5

    # No vertex sits on the original corner any more.
    assert np.linalg.norm(cut.vertices - np.array([0.0, 0.0]), axis=1).min() > 7.9


def test_chamfered_corner_is_found_and_absent_when_uncut():
    assert chamfered_corner(plain_square()) is None
    assert not is_chamfered(plain_square())

    cut = chamfer(plain_square(), 8.0)
    assert is_chamfered(cut)
    assert np.allclose(chamfered_corner(cut), [0.0, 0.0])


@pytest.mark.parametrize("corner,expected", [
    ("min", [0.0, 0.0]),
    ("maxx", [150.0, 0.0]),
    ("maxy", [0.0, 150.0]),
    ("max", [150.0, 150.0]),
])
def test_chamfer_can_target_any_corner(corner, expected):
    cut = chamfer(plain_square(), 8.0, corner=corner)
    assert np.allclose(chamfered_corner(cut), expected)


def test_chamfer_survives_the_face_flip():
    """The cut is boundary geometry, so mirroring carries it across for free."""
    cut = chamfer(plain_square(), 8.0)
    assert np.allclose(chamfered_corner(back_face(cut)), [150.0, 0.0])
    assert np.allclose(chamfered_corner(back_face(back_face(cut))), [0.0, 0.0])


@pytest.mark.parametrize("name", sorted(PILOT_DESIGNS))
def test_chamfering_a_pilot_design_keeps_it_foldable(name):
    """Cutting a corner removes paper, never a crease, so validity is untouched."""
    original = PILOT_DESIGNS[name]()
    cut = chamfer(original, 8.0)
    assert verify(cut).valid
    assert len(cut.interior_vertices()) == len(original.interior_vertices())


def test_chamfer_refuses_to_cut_through_a_crease():
    """Silently trimming a fold would corrupt the ground truth."""
    square = plain_square()
    vertices = np.vstack([square.vertices, [[75.0, 75.0]]])
    edges = np.vstack([square.edges, [[0, 4]]])
    with_crease = CreasePattern(vertices, edges, np.array(list("BBBBM")))

    with pytest.raises(ValueError, match="carries a crease"):
        chamfer(with_crease, 8.0)


def test_chamfer_refuses_an_oversized_cut():
    with pytest.raises(ValueError, match="exceeds"):
        chamfer(plain_square(), 200.0)


def test_detection_finds_the_cut_in_a_render():
    cut = chamfer(pleat(6), 10.0)
    for rotation_deg in (0.0, 23.0, 137.0):
        image, _ = render(
            cut, ScanStyle(size_px=700, rotation_deg=rotation_deg), np.random.default_rng(2)
        )
        outline = detect_sheet_outline(image)
        assert outline.chamfer_index is not None, f"missed the cut at {rotation_deg} deg"
        assert len(outline.corners) == 4


def test_detection_reports_no_cut_on_a_plain_sheet():
    image, _ = render(pleat(6), ScanStyle(size_px=700, rotation_deg=17.0), np.random.default_rng(2))
    assert detect_sheet_outline(image).chamfer_index is None


@pytest.mark.parametrize("rotation_deg", [0.0, 23.0, 137.0, 250.0])
def test_the_cut_resolves_the_symmetric_miura(rotation_deg):
    """The case the fiducial exists for.

    Without it registration picks between two tied orientations and lands
    hundreds of pixels out half the time, with every MV label inverted and a
    healthy-looking score. With it there is only one pose to pick.
    """
    cut = chamfer(miura(4, 4), 8.0)
    style = ScanStyle(size_px=700, rotation_deg=rotation_deg)
    image, corners = render(cut, style, np.random.default_rng(11))

    result = register(image, cut)
    assert result.used_fiducial
    assert result.confident

    error = np.linalg.norm(result.project(cut) - _project(cut, corners), axis=1).max()
    assert error < 8.0, f"aligned {error:.0f}px out"


def test_without_the_cut_the_miura_stays_ambiguous():
    """Contrast case: the chamfer is what changes the outcome, not luck."""
    cp = miura(4, 4)
    image, _ = render(cp, ScanStyle(size_px=700, rotation_deg=23.0), np.random.default_rng(11))

    result = register(image, cp)
    assert not result.used_fiducial
    assert result.margin == pytest.approx(1.0, abs=0.05)
    assert not result.confident


def test_asymmetric_designs_still_register_with_a_cut():
    """The fiducial must not make the ordinary case worse."""
    cut = chamfer(PILOT_DESIGNS["d03_vertex6"](), 8.0)
    style = ScanStyle(size_px=700, rotation_deg=41.0)
    image, corners = render(cut, style, np.random.default_rng(5))

    result = register(image, cut)
    assert result.used_fiducial and result.confident
    error = np.linalg.norm(result.project(cut) - _project(cut, corners), axis=1).max()
    assert error < 8.0


def test_chamfer_scales_with_the_sheet():
    cut = chamfer(pleat(4, size=SHEET_MM), 12.0)
    polygon = cut.boundary_polygon()
    edge_lengths = np.linalg.norm(
        cut.vertices[np.roll(polygon, -1)] - cut.vertices[polygon], axis=1
    )
    # The cut spans the hypotenuse between two points 12mm along each edge.
    assert np.isclose(edge_lengths.min(), 12.0 * np.sqrt(2), atol=0.1)
