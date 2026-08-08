"""Tests for the exact single-vertex solver and the random pattern families.

Synthetic data is only worth pre-training on if it is genuinely foldable, so
validity is asserted on samples from every family rather than assumed.
"""

import itertools

import numpy as np
import pytest

from origamicp.generate.random_cp import (
    FAMILIES,
    random_corrugation,
    random_design,
    random_pleat,
    random_sector_angles,
    random_single_vertex,
)
from origamicp.verify import big_little_big_violations, kawasaki_residual, maekawa_defect, verify
from origamicp.verify.single_vertex import is_flat_foldable, valid_assignments


# --------------------------------------------------------------------------
# exact single-vertex foldability
# --------------------------------------------------------------------------


def test_crimping_agrees_with_the_local_conditions_at_degree_four():
    """Degree four is the case with a known exact characterisation.

    Kawasaki plus Maekawa plus big-little-big is necessary *and* sufficient
    there, so it pins down the crimp search on a case we can check independently
    -- including the tie handling, which is where the textbook statement stops.
    """
    rng = np.random.default_rng(0)
    mismatches = 0
    for _ in range(300):
        angles = np.empty(4)
        angles[0::2] = rng.dirichlet([2.5, 2.5]) * 180
        angles[1::2] = rng.dirichlet([2.5, 2.5]) * 180
        angles = np.deg2rad(angles)

        for mv in itertools.product("MV", repeat=4):
            mv = np.array(mv)
            by_conditions = (
                abs(maekawa_defect(mv)) == 2
                and kawasaki_residual(angles) < 1e-9
                and not big_little_big_violations(angles, mv, 1e-9)
            )
            mismatches += is_flat_foldable(angles, mv) != by_conditions
    assert mismatches == 0


@pytest.mark.parametrize("degree", [4, 6, 8])
def test_equal_angle_vertices_admit_exactly_the_maekawa_assignments(degree):
    """With no sector smaller than its neighbours there is nothing to obstruct.

    Counts: C(2n, n+1) * 2 -- 8, 30 and 112 for degrees four, six and eight.
    """
    from math import comb

    found = valid_assignments(np.full(degree, 2 * np.pi / degree))
    assert len(found) == 2 * comb(degree, degree // 2 + 1)
    assert all(abs(maekawa_defect(np.array(list(m)))) == 2 for m in found)


def test_maekawa_is_necessary():
    """Nothing the solver accepts may violate the three-to-one split."""
    rng = np.random.default_rng(1)
    for _ in range(40):
        angles = np.deg2rad(random_sector_angles(rng, 6))
        for mv in valid_assignments(angles):
            assert abs(maekawa_defect(np.array(list(mv)))) == 2


def test_bad_geometry_and_labels_are_rejected():
    right = np.full(4, np.pi / 2)
    assert not is_flat_foldable(right, np.array(list("MMMM")))  # Maekawa
    assert not is_flat_foldable(np.deg2rad([60, 120, 60, 120]), np.array(list("MMMV")))
    assert not is_flat_foldable(np.deg2rad([120, 120, 120]), np.array(list("MVM")))  # odd
    assert not is_flat_foldable(right, np.array(list("MMMU")))  # unassigned crease
    assert not is_flat_foldable(right, np.array(list("MMM")))  # length mismatch


def test_big_little_big_is_enforced_by_the_solver():
    """30/100/150/80 passes Maekawa and Kawasaki; only the little sector rejects it."""
    angles = np.deg2rad([30, 100, 150, 80])
    assert not is_flat_foldable(angles, np.array(list("MMMV")))
    assert is_flat_foldable(angles, np.array(list("MVMM")))


# --------------------------------------------------------------------------
# random families
# --------------------------------------------------------------------------


def test_random_sector_angles_satisfy_kawasaki():
    rng = np.random.default_rng(2)
    for degree in (4, 6, 8, 10):
        for _ in range(50):
            angles = random_sector_angles(rng, degree)
            assert len(angles) == degree
            assert np.isclose(angles.sum(), 360.0)
            assert kawasaki_residual(np.deg2rad(angles)) < 1e-9
            assert angles.min() >= 16.0


def test_random_sector_angles_rejects_impossible_requests():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="even"):
        random_sector_angles(rng, 5)
    with pytest.raises(ValueError, match="cannot fit"):
        random_sector_angles(rng, 30)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_every_random_family_produces_valid_patterns(family):
    rng = np.random.default_rng(4)
    for _ in range(25):
        cp = FAMILIES[family](rng)
        report = verify(cp)
        assert report.valid, f"{family}: {report.summary()}"


def test_random_single_vertex_is_exactly_foldable():
    """Not just locally valid -- the crimp test must accept it too."""
    rng = np.random.default_rng(5)
    for _ in range(30):
        cp = random_single_vertex(rng)
        centre = cp.interior_vertices()[0]
        assert is_flat_foldable(cp.sector_angles(centre), cp.mv_around(centre))


def test_random_single_vertex_is_off_centre_and_rotated():
    """Both are needed or registration would be ambiguous on synthetic data."""
    rng = np.random.default_rng(6)
    centres, first_angles = [], []
    for _ in range(20):
        cp = random_single_vertex(rng)
        centres.append(cp.vertices[cp.interior_vertices()[0]])
        _, theta = cp.sorted_creases(cp.interior_vertices()[0])
        first_angles.append(theta[0])
    assert np.std(centres, axis=0).min() > 1.0
    assert np.std(first_angles) > 0.3


def test_random_designs_cover_every_family():
    rng = np.random.default_rng(7)
    seen = {random_design(rng)[0] for _ in range(120)}
    assert seen == set(FAMILIES)


def test_corrugation_row_geometry_varies():
    """Rows may vary freely; columns may not, because Kawasaki depends on them."""
    rng = np.random.default_rng(8)
    cp = random_corrugation(rng)
    assert verify(cp).valid
    xs = np.unique(np.round(cp.vertices[:, 0], 6))
    assert np.allclose(np.diff(xs), np.diff(xs)[0]), "column spacing must stay uniform"


def test_pleat_has_uneven_panels_but_stays_valid():
    rng = np.random.default_rng(9)
    cp = random_pleat(rng)
    assert verify(cp).valid
    assert cp.interior_vertices() == []
