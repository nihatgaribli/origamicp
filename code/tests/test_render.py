"""Tests for the synthetic scan renderer.

The renderer's job is to reproduce one cue correctly: which side of a crease
catches the light. If that is wrong or missing, a model pre-trained on this data
learns nothing transferable, and no downstream metric would reveal it -- the
synthetic numbers would look fine. Hence the physics is asserted here directly.
"""

import cv2
import numpy as np
import pytest

from origamicp.capture import register
from origamicp.core import CreasePattern
from origamicp.render import BACKLIT, PHOTOMETRIC, ScanStyle, photometric_stack, render

SIZE = 600
CREASE_FRACTIONS = {"M": 1 / 3, "V": 2 / 3}


def two_crease_sheet() -> CreasePattern:
    """A sheet with one vertical mountain and one vertical valley."""
    vertices = np.array(
        [[0, 0], [100, 0], [100, 100], [0, 100], [33, 0], [33, 100], [66, 0], [66, 100]],
        dtype=np.float64,
    )
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [6, 7]])
    return CreasePattern(vertices, edges, np.array(list("BBBBMV")))


def clean_style(**overrides) -> ScanStyle:
    """Deterministic geometry with the stochastic dressing switched off."""
    base = dict(
        size_px=SIZE,
        rotation_deg=0.0,
        undulation=0.0,
        fiber_noise=0.0,
        albedo_variation=0.0,
        sensor_noise=0.0,
        crease_wobble_px=0.0,
        crease_strength_jitter=0.0,
    )
    return ScanStyle(**{**base, **overrides})


def grey(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)


def crease_x(fraction: float, style: ScanStyle) -> int:
    margin = style.margin_frac * style.size_px
    return int(round(margin + (style.size_px - 2 * margin) * fraction))


def shading_asymmetry(image: np.ndarray, fraction: float, style: ScanStyle) -> float:
    """Brightness on the +x side of a crease minus the -x side."""
    row = grey(image)[style.size_px // 2]
    x = crease_x(fraction, style)
    return row[x + 3 : x + 10].mean() - row[x - 10 : x - 3].mean()


def test_mountain_and_valley_shade_oppositely():
    """The whole task rests on this sign difference."""
    style = clean_style(light_azimuth_deg=0.0)
    image, _ = render(two_crease_sheet(), style, np.random.default_rng(0))

    mountain = shading_asymmetry(image, CREASE_FRACTIONS["M"], style)
    valley = shading_asymmetry(image, CREASE_FRACTIONS["V"], style)

    assert mountain > 2.0, f"mountain asymmetry too weak: {mountain:+.1f}"
    assert valley < -2.0, f"valley asymmetry too weak: {valley:+.1f}"


@pytest.mark.parametrize("fraction", CREASE_FRACTIONS.values())
def test_flipping_the_light_inverts_the_cue(fraction):
    """Light direction is part of the label, which is why the manifest records it."""
    sheet = two_crease_sheet()
    near = clean_style(light_azimuth_deg=0.0)
    far = clean_style(light_azimuth_deg=180.0)

    a = shading_asymmetry(render(sheet, near, np.random.default_rng(0))[0], fraction, near)
    b = shading_asymmetry(render(sheet, far, np.random.default_rng(0))[0], fraction, far)

    assert a * b < 0, f"cue did not invert: {a:+.1f} then {b:+.1f}"


def test_flat_paper_does_not_clip():
    """Rendering paper at full white silently destroys half of the MV cue.

    Every crease has a lit side and a shadowed side. If flat paper sits at 255
    the lit side clips away, valleys keep their shadow, mountains lose their
    highlight, and the asymmetry test above passes for valleys only.
    """
    style = clean_style()
    image = grey(render(two_crease_sheet(), style, np.random.default_rng(0))[0])
    sheet = image[image > style.background + 40]

    assert (sheet >= 255).sum() == 0, "highlights are clipping"
    assert sheet.max() < 250
    assert sheet.min() > 120  # nor crushed into black


def test_backlit_carries_no_mv_information():
    """Transmission brightens both fold directions equally, by construction."""
    style = clean_style(mode=BACKLIT)
    row = grey(render(two_crease_sheet(), style, np.random.default_rng(0))[0])[SIZE // 2]

    peaks = []
    for fraction in CREASE_FRACTIONS.values():
        x = crease_x(fraction, style)
        peaks.append(row[x - 6 : x + 7].max())  # the crease centre is sub-pixel

    assert abs(peaks[0] - peaks[1]) < 2.0, f"backlit leaks MV: {peaks}"
    assert peaks[0] > row[SIZE // 2] + 5  # creases are still visible


def test_boundary_edges_are_not_creases():
    """The sheet edge is a cut. Rendering it as a fold would teach a phantom."""
    style = clean_style()
    sheet = two_crease_sheet()
    flat = CreasePattern(sheet.vertices, sheet.edges[:4], sheet.assignment[:4])

    image = grey(render(flat, style, np.random.default_rng(0))[0])
    interior = image[100:-100, 100:-100]
    assert np.ptp(interior) < 3.0, "found relief on a sheet with no creases"


def test_render_is_reproducible():
    style = clean_style(crease_wobble_px=0.8, undulation=0.9, sensor_noise=0.004)
    a, _ = render(two_crease_sheet(), style, np.random.default_rng(7))
    b, _ = render(two_crease_sheet(), style, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_photometric_stack_moves_only_the_light():
    """Photometric stereo needs the camera and paper pinned across frames."""
    frames = photometric_stack(two_crease_sheet(), style=clean_style(), seed=3)
    assert len(frames) == 8

    azimuths = [a for a, _ in frames]
    assert azimuths == [0, 45, 90, 135, 180, 225, 270, 315]

    masks = [grey(img) > 120 for _, img in frames]
    for mask in masks[1:]:
        assert np.array_equal(mask.shape, masks[0].shape)
        # The sheet occupies the same pixels; only shading differs.
        assert (mask == masks[0]).mean() > 0.99

    profiles = [grey(img)[SIZE // 2] for _, img in frames]
    assert not np.allclose(profiles[0], profiles[4]), "opposite lights look identical"


@pytest.mark.parametrize("rotation_deg", [0.0, 23.0, 61.0])
def test_registration_recovers_a_synthetic_render(rotation_deg):
    """Closes the loop: the capture pipeline must work on generated data too."""
    from origamicp.generate.designs import PILOT_DESIGNS

    cp = PILOT_DESIGNS["d03_vertex6"]()
    style = ScanStyle(size_px=700, rotation_deg=rotation_deg)
    image, corners = render(cp, style, np.random.default_rng(11))

    result = register(image, cp)
    assert result.confident, f"score={result.score:.3f} margin={result.margin:.2f}"

    projected = result.project(cp)
    lo, hi = projected.min(axis=0), projected.max(axis=0)
    assert np.allclose(lo, corners.min(axis=0), atol=6)
    assert np.allclose(hi, corners.max(axis=0), atol=6)


def test_rotated_sheet_stays_inside_the_frame():
    """A clipped sheet loses its corners, and registration then has nothing to fit."""
    for rotation_deg in (0.0, 23.0, 45.0, 61.0, 137.0):
        style = ScanStyle(size_px=700, rotation_deg=rotation_deg)
        _, corners = render(two_crease_sheet(), style, np.random.default_rng(0))
        assert corners.min() > 0 and corners.max() < style.size_px


def test_miura_at_180_degrees_is_genuinely_ambiguous():
    """A documented limitation, not a bug -- and a hazard for the real dataset.

    Rotating a Miura sheet by 180 degrees maps its geometry onto itself while
    swapping every mountain and valley. The registrar scores on an unsigned
    ridge response by design, so it cannot break that tie and correctly reports
    the sheet as ambiguous rather than guessing.

    Guessing would be the dangerous outcome: half the corrugation sheets would
    come back with every MV label inverted, and no downstream check would catch
    it. Symmetric designs need a physical fiducial on the sheet.
    """
    from origamicp.generate.designs import PILOT_DESIGNS

    cp = PILOT_DESIGNS["d05_miura_small"]()
    image, _ = render(cp, ScanStyle(size_px=700, rotation_deg=23.0), np.random.default_rng(11))

    result = register(image, cp)
    assert result.score > 0.05, "the pattern is found; only its orientation is unclear"
    assert result.margin == pytest.approx(1.0, abs=0.05)
    assert not result.confident


def test_photometric_mode_vignettes():
    style = clean_style(mode=PHOTOMETRIC)
    image = grey(render(two_crease_sheet(), style, np.random.default_rng(0))[0])
    centre = image[SIZE // 2 - 20 : SIZE // 2 + 20, SIZE // 2 - 20 : SIZE // 2 + 20].mean()
    corner = image[60:100, 60:100].mean()
    assert centre > corner


def test_default_style_keeps_the_mv_cue_above_the_paper_texture():
    """Difficulty calibration, locked in.

    If the crease signal drops near the texture level the renders become
    unlearnable, and nothing downstream would say so -- the synthetic metrics
    would simply be bad without explaining why. This is the guard.
    """
    from origamicp.generate.designs import PILOT_DESIGNS
    from origamicp.render.scan import measure_crease_snr

    for name in ("d03_vertex6", "d05_miura_small"):
        cp = PILOT_DESIGNS[name]()
        signal, noise = measure_crease_snr(
            cp, ScanStyle(size_px=600, rotation_deg=0.0), np.random.default_rng(4)
        )
        assert signal / noise > 2.0, f"{name}: SNR {signal / noise:.2f} too low"


def test_snr_tracks_the_difficulty_parameters():
    """A faint crease on rough paper must measure as harder than a sharp one."""
    from origamicp.generate.designs import PILOT_DESIGNS
    from origamicp.render.scan import measure_crease_snr

    cp = PILOT_DESIGNS["d05_miura_small"]()
    ratios = []
    for crease_height, undulation, fiber in [(0.5, 1.0, 0.07), (1.5, 0.2, 0.02)]:
        signal, noise = measure_crease_snr(
            cp,
            ScanStyle(
                size_px=600,
                rotation_deg=0.0,
                crease_height=crease_height,
                undulation=undulation,
                fiber_noise=fiber,
            ),
            np.random.default_rng(4),
        )
        ratios.append(signal / noise)

    assert ratios[0] < ratios[1], f"difficulty did not order: {ratios}"
    assert ratios[0] > 1.0, "the hard end must stay solvable"
