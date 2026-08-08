"""Tests for the capture pipeline: face flipping, manifest hygiene, registration."""

import cv2
import numpy as np
import pytest

from origamicp.capture import (
    BACK,
    FRONT,
    PHOTOMETRIC,
    SCANNER,
    CaptureRecord,
    Manifest,
    back_face,
    detect_sheet_quad,
    flip_assignment,
    mv_flip_consistency,
    register,
)
from origamicp.core import CreasePattern
from origamicp.generate import single_vertex
from origamicp.verify import verify


# --------------------------------------------------------------------------
# front / back face
# --------------------------------------------------------------------------


def test_flip_assignment_only_touches_folds():
    flipped = flip_assignment(np.array(list("MVBFU")))
    assert list(flipped) == ["V", "M", "B", "F", "U"]


def test_back_face_is_an_involution():
    cp = single_vertex([30, 100, 150, 80], "MVMM")
    twice = back_face(back_face(cp))
    assert np.allclose(twice.vertices, cp.vertices)
    assert np.array_equal(twice.assignment, cp.assignment)


def test_back_face_preserves_validity():
    """The physics is the same sheet, so the geometry checks must agree.

    Mirroring reverses the cyclic order of the sectors and flipping negates the
    Maekawa defect; both conditions are stated up to sign, so validity holds.
    This is a real check on the verifier, not just on the flip.
    """
    for angles, mv in [
        ([30, 100, 150, 80], "MVMM"),  # valid
        ([30, 100, 150, 80], "MMMV"),  # big-little-big violation
        ([60, 120, 60, 120], "MMMV"),  # Kawasaki violation
        ([90, 90, 90, 90], "MMMM"),  # Maekawa violation
    ]:
        cp = single_vertex(angles, mv)
        front, back = verify(cp), verify(back_face(cp))
        assert front.valid == back.valid, f"{angles} {mv}"
        assert front.vertex_validity_rate == back.vertex_validity_rate


def test_mv_flip_consistency_scores():
    front = np.array(list("MMVV"))
    assert mv_flip_consistency(front, np.array(list("VVMM"))) == 1.0
    assert mv_flip_consistency(front, np.array(list("MMVV"))) == 0.0
    assert mv_flip_consistency(front, np.array(list("MVVM"))) == 0.5  # 2 of 4 invert
    assert mv_flip_consistency(front, np.array(list("VVVM"))) == 0.75


def test_mv_flip_consistency_ignores_uncommitted_edges():
    """Hedging with U neither helps nor hurts, so coverage must be reported."""
    score = mv_flip_consistency(np.array(list("MUUV")), np.array(list("VUUM")))
    assert score == 1.0
    assert np.isnan(mv_flip_consistency(np.array(list("UU")), np.array(list("UU"))))


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


def _four_scans(tmp_path, sheet_id="s001", design_id="d01", suffix=".png", dpi=600):
    records = []
    for face in (FRONT, BACK):
        for rotation in (0, 90):
            name = f"{sheet_id}_{face}_{rotation}{suffix}"
            (tmp_path / name).write_bytes(b"stub")
            records.append(
                CaptureRecord(
                    sheet_id=sheet_id,
                    design_id=design_id,
                    image_path=name,
                    face=face,
                    rotation_deg=rotation,
                    dpi=dpi,
                    folder_name="nihat",
                )
            )
    return records


def test_complete_sheet_passes(tmp_path):
    manifest = Manifest(_four_scans(tmp_path))
    assert [i for i in manifest.validate(tmp_path) if i.severity == "error"] == []


def test_missing_back_scan_is_an_error(tmp_path):
    records = [r for r in _four_scans(tmp_path) if r.face != BACK or r.rotation_deg != 90]
    errors = [i for i in Manifest(records).validate(tmp_path) if i.severity == "error"]
    assert any("face=back rotation=90" in i.message for i in errors)


def test_jpeg_is_rejected(tmp_path):
    manifest = Manifest(_four_scans(tmp_path, suffix=".jpg"))
    errors = [i for i in manifest.validate(tmp_path) if i.severity == "error"]
    assert any("lossy" in i.message for i in errors)


def test_low_dpi_is_rejected(tmp_path):
    manifest = Manifest(_four_scans(tmp_path, dpi=300))
    assert any("dpi 300" in i.message for i in manifest.validate(tmp_path))


def test_missing_file_is_reported(tmp_path):
    records = _four_scans(tmp_path)
    (tmp_path / records[0].image_path).unlink()
    errors = [i for i in Manifest(records).validate(tmp_path) if i.severity == "error"]
    assert any("missing image" in i.message for i in errors)


def test_design_leaking_across_splits_is_an_error(tmp_path):
    records = _four_scans(tmp_path, sheet_id="s001", design_id="d01")
    records += _four_scans(tmp_path, sheet_id="s002", design_id="d01")
    for r in records[:4]:
        r.split = "train"
    for r in records[4:]:
        r.split = "test"

    errors = [i for i in Manifest(records).validate(tmp_path) if i.severity == "error"]
    assert any("leaks across splits" in i.message for i in errors)


def test_photometric_needs_azimuth(tmp_path):
    (tmp_path / "p.png").write_bytes(b"stub")
    record = CaptureRecord(
        sheet_id="s001", design_id="d01", image_path="p.png", modality=PHOTOMETRIC
    )
    errors = [i for i in Manifest([record]).validate(tmp_path) if i.severity == "error"]
    assert any("light_azimuth_deg" in i.message for i in errors)


def test_manifest_round_trips(tmp_path):
    manifest = Manifest(_four_scans(tmp_path))
    manifest.records[0].notes = "corner slightly torn"
    manifest.records[0].light_azimuth_deg = 45.0
    path = tmp_path / "manifest.csv"
    manifest.save(path)

    restored = Manifest.load(path)
    assert len(restored) == 4
    assert restored.records[0].notes == "corner slightly torn"
    assert restored.records[0].light_azimuth_deg == 45.0
    assert restored.records[0].dpi == 600
    assert restored.records[0].modality == SCANNER
    assert "4 scans | 1 sheets | 1 designs" in restored.summary()


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


def asymmetric_cp() -> CreasePattern:
    """A unit sheet whose creases have no rotational or mirror symmetry."""
    vertices = np.array(
        [[0, 0], [1, 0], [1, 1], [0, 1], [0.18, 0.22], [0.82, 0.34], [0.36, 0.88]],
        dtype=np.float64,
    )
    edges = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [4, 6], [4, 1]], dtype=np.int64
    )
    assignment = np.array(list("BBBBMVMV"), dtype="<U1")
    return CreasePattern(vertices, edges, assignment)


def render_scan(cp, size=600, margin=60, rotation_deg=0.0, mirror=False):
    """Fake a scan: a bright sheet on dark backing with faint crease lines.

    Good enough to exercise corner finding and orientation disambiguation; the
    real thing has texture and noise, which only makes the ridge response
    stronger, not weaker.
    """
    lo, hi = cp.vertices.min(axis=0), cp.vertices.max(axis=0)
    unit = (cp.vertices - lo) / (hi - lo)
    if mirror:
        unit = np.column_stack([1.0 - unit[:, 0], unit[:, 1]])

    theta = np.deg2rad(rotation_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    unit = (unit - 0.5) @ rot.T + 0.5

    span = size - 2 * margin
    px = np.column_stack([margin + unit[:, 0] * span, margin + (1 - unit[:, 1]) * span])

    image = np.full((size, size), 28, dtype=np.uint8)
    corners = px[:4].astype(np.int32)
    cv2.fillConvexPoly(image, corners, 236)
    for (a, b), kind in zip(cp.edges, cp.assignment):
        if kind == "B":
            continue
        shade = 214 if kind == "M" else 252  # a crease is a faint highlight or shadow
        cv2.line(image, tuple(px[a].astype(int)), tuple(px[b].astype(int)), shade, 3)
    image = cv2.GaussianBlur(image, (0, 0), 1.2)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), px


def test_detect_sheet_quad_finds_the_corners():
    cp = asymmetric_cp()
    image, px = render_scan(cp)
    quad = detect_sheet_quad(image)

    # Same four points, in some cyclic order.
    distances = np.linalg.norm(quad[:, None, :] - px[None, :4, :], axis=2)
    assert distances.min(axis=1).max() < 5.0


@pytest.mark.parametrize("rotation_deg", [0.0, 90.0, 180.0, 270.0])
@pytest.mark.parametrize("mirror", [False, True])
def test_register_recovers_every_orientation(rotation_deg, mirror):
    """The operator's recorded orientation is never trusted; alignment decides."""
    cp = asymmetric_cp()
    image, px = render_scan(cp, rotation_deg=rotation_deg, mirror=mirror)

    result = register(image, cp)
    error = np.linalg.norm(result.project(cp) - px, axis=1)

    assert error.max() < 6.0, f"max error {error.max():.1f}px"
    assert result.confident, f"score={result.score:.3f} margin={result.margin:.2f}"


def test_register_survives_a_tilted_sheet():
    cp = asymmetric_cp()
    image, px = render_scan(cp, rotation_deg=7.0)
    result = register(image, cp)
    assert np.linalg.norm(result.project(cp) - px, axis=1).max() < 8.0


def test_symmetric_pattern_is_reported_as_ambiguous():
    """A four-fold symmetric sheet genuinely has no recoverable orientation.

    The right behaviour is to say so, not to pick one and look confident.
    """
    vertices = np.array(
        [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], dtype=np.float64
    )
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 0], [4, 1], [4, 2], [4, 3]])
    cp = CreasePattern(vertices, edges, np.array(list("BBBBMMMV"), dtype="<U1"))

    image, _ = render_scan(cp)
    result = register(image, cp)
    assert result.margin < 1.15
    assert not result.confident
