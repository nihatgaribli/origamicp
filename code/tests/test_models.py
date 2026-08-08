"""Tests for targets, dataset, network and metrics.

The recurring risk in this layer is a label that is subtly wrong -- shifted by a
pose error, or not flipped on the back face. Training would still converge and
the numbers would still look plausible, so the labels are checked directly
against the geometry rather than by eye.
"""

import numpy as np
import pytest
import torch

from origamicp.capture import back_face
from origamicp.core import CreasePattern
from origamicp.generate.designs import PILOT_DESIGNS
from origamicp.models import (
    Counts,
    CreaseUNet,
    SyntheticCreaseDataset,
    build_targets,
    collate,
    light_channels,
    losses,
    project_to_pixels,
)
from origamicp.render import ScanStyle, render


def rendered(name="d05_miura_small", size_px=256, rotation_deg=17.0, seed=3):
    cp = PILOT_DESIGNS[name]()
    style = ScanStyle(size_px=size_px, rotation_deg=rotation_deg)
    image, corners = render(cp, style, np.random.default_rng(seed))
    return cp, image, corners


# --------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------


def test_targets_have_the_expected_shapes_and_values():
    cp, _, corners = rendered()
    targets = build_targets(cp, corners, 256)

    assert set(targets) == {"crease", "mv", "sheet"}
    for mask in targets.values():
        assert mask.shape == (256, 256)
    assert set(np.unique(targets["crease"])) <= {0, 1}
    assert set(np.unique(targets["mv"])) <= {0, 1, 2}
    assert targets["crease"].any() and targets["sheet"].any()


def test_crease_target_sits_on_the_projected_geometry():
    """A pose error here would shift every label and never announce itself."""
    cp, _, corners = rendered()
    targets = build_targets(cp, corners, 256)
    pixels = project_to_pixels(cp, corners)

    hits = 0
    for (a, b), kind in zip(cp.edges, cp.assignment):
        if kind == "B":
            continue
        midpoint = np.round((pixels[a] + pixels[b]) / 2).astype(int)
        if not (0 <= midpoint[0] < 256 and 0 <= midpoint[1] < 256):
            continue
        hits += targets["crease"][midpoint[1], midpoint[0]] == 1
    assert hits >= 0.9 * sum(1 for k in cp.assignment if k != "B")


def test_boundary_edges_are_not_labelled_as_creases():
    cp, _, corners = rendered()
    targets = build_targets(cp, corners, 256)
    pixels = project_to_pixels(cp, corners)

    for (a, b), kind in zip(cp.edges, cp.assignment):
        if kind != "B":
            continue
        midpoint = np.round((pixels[a] + pixels[b]) / 2).astype(int)
        x = np.clip(midpoint[0], 2, 253)
        y = np.clip(midpoint[1], 2, 253)
        # Allow a couple of pixels for creases that legitimately end there.
        assert targets["crease"][y - 1 : y + 2, x - 1 : x + 2].sum() <= 6


def test_mountains_and_valleys_get_distinct_labels():
    cp, _, corners = rendered()
    targets = build_targets(cp, corners, 256)
    labels = set(np.unique(targets["mv"][targets["crease"] == 1]))
    assert labels == {1, 2}


def test_creases_are_clipped_to_the_sheet():
    cp, _, corners = rendered()
    targets = build_targets(cp, corners, 256)
    assert not (targets["crease"] & (1 - targets["sheet"])).any()


def test_light_channels_encode_direction_and_absence():
    zeros = light_channels(None, 8)
    assert zeros.shape == (2, 8, 8) and not zeros.any()

    east = light_channels(0.0, 8, elevation_deg=0.0)
    assert np.allclose(east[0], 1.0) and np.allclose(east[1], 0.0)

    north = light_channels(90.0, 8, elevation_deg=0.0)
    assert np.allclose(north[0], 0.0, atol=1e-6) and np.allclose(north[1], 1.0)


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory):
    """A tiny generated set, built through the real script."""
    from scripts.make_synthetic import main as make

    root = tmp_path_factory.mktemp("synth")
    assert make(["--n", "12", "--out", str(root), "--size-px", "128", "--seed", "1"]) == 0
    return root


def test_dataset_items_are_consistent(synth_root):
    dataset = SyntheticCreaseDataset(synth_root, None, crop=None, jitter=False)
    item = dataset[0]

    assert item["image"].shape == (3, 128, 128)
    assert item["crease"].shape == (128, 128)
    assert item["mv"].dtype == torch.int64
    # Channel 0 is the scan; channels 1 and 2 are cos and sin of the light
    # azimuth and are legitimately negative. Asserting one range over all three
    # passed only while the sampled azimuth happened to fall in a quadrant where
    # both were positive.
    scan = item["image"][0]
    assert scan.min() >= -0.01 and scan.max() <= 1.01
    light = item["image"][1:]
    assert light.min() >= -1.01 and light.max() <= 1.01
    # Labels only exist where a crease is, and vice versa.
    assert torch.equal((item["mv"] > 0).float(), item["crease"])


def test_back_face_items_use_flipped_labels(synth_root):
    """The back scan is a different surface; reusing the front labels would
    invert every mountain and valley in half the dataset."""
    dataset = SyntheticCreaseDataset(synth_root, None, crop=None, jitter=False)
    by_id = {}
    for index, row in enumerate(dataset.rows):
        by_id.setdefault(row["design_id"], {})[row["face"]] = index

    differed = 0
    for faces in by_id.values():
        if {"front", "back"} - set(faces):
            continue
        row = dataset.rows[faces["back"]]
        cp = CreasePattern.from_fold(synth_root / row["fold_path"])
        corners = np.fromstring(row["corners"], sep=" ").reshape(4, 2)

        expected = build_targets(back_face(cp), corners, 128)
        assert np.array_equal(dataset[faces["back"]]["mv"].numpy(), expected["mv"])
        differed += not np.array_equal(expected["mv"], build_targets(cp, corners, 128)["mv"])

    # Some patterns map onto themselves under mirror-plus-swap -- a symmetric
    # pleat does -- so the flip is not required to change every design. It must
    # change most of them, or the flip is not being applied at all.
    assert differed >= len(by_id) // 2


def test_backlit_samples_are_excluded_from_mv_supervision(synth_root):
    dataset = SyntheticCreaseDataset(synth_root, None, crop=None, jitter=False)
    for index, row in enumerate(dataset.rows):
        expected = 0.0 if row["modality"] == "backlit" else 1.0
        assert float(dataset[index]["mv_valid"]) == expected
        if row["modality"] == "backlit":
            assert not dataset[index]["image"][1:].any()  # no light direction


def test_cropping_and_split_filtering(synth_root):
    cropped = SyntheticCreaseDataset(synth_root, None, crop=64, jitter=False)
    assert cropped[0]["image"].shape == (3, 64, 64)

    train = SyntheticCreaseDataset(synth_root, "train", crop=None, jitter=False)
    assert all(r["split"] == "train" for r in train.rows)
    with pytest.raises(ValueError, match="no rows"):
        SyntheticCreaseDataset(synth_root, "nonexistent")


def test_splits_do_not_share_designs(synth_root):
    """The leakage check that matters: memorising a design must not be rewarded."""
    seen = {}
    for split in ("train", "val", "test"):
        try:
            dataset = SyntheticCreaseDataset(synth_root, split, crop=None)
        except ValueError:
            continue
        for row in dataset.rows:
            assert seen.setdefault(row["design_id"], split) == split


def test_collate_keeps_design_ids_as_strings(synth_root):
    dataset = SyntheticCreaseDataset(synth_root, None, crop=64, jitter=False)
    batch = collate([dataset[0], dataset[1]])
    assert batch["image"].shape == (2, 3, 64, 64)
    assert isinstance(batch["design_id"], list) and len(batch["design_id"]) == 2


# --------------------------------------------------------------------------
# network and losses
# --------------------------------------------------------------------------


def test_unet_output_shapes():
    model = CreaseUNet(in_channels=3, width=8, depth=3)
    outputs = model(torch.randn(2, 3, 64, 64))
    assert outputs["crease"].shape == (2, 64, 64)
    assert outputs["mv"].shape == (2, 2, 64, 64)


def test_unet_handles_sizes_that_are_not_powers_of_two():
    model = CreaseUNet(in_channels=3, width=8, depth=3)
    assert model(torch.randn(1, 3, 70, 70))["crease"].shape == (1, 70, 70)


def fake_batch(mv_valid=1.0, size=32):
    crease = torch.zeros(2, size, size)
    crease[:, 10:13, :] = 1.0
    mv = torch.zeros(2, size, size, dtype=torch.long)
    mv[:, 10:13, :] = 1
    return {
        "crease": crease,
        "mv": mv,
        "sheet": torch.ones(2, size, size),
        "mv_valid": torch.full((2,), mv_valid),
    }


def test_losses_are_finite_and_positive():
    batch = fake_batch()
    outputs = {
        "crease": torch.zeros(2, 32, 32, requires_grad=True),
        "mv": torch.zeros(2, 2, 32, 32, requires_grad=True),
    }
    loss = losses(outputs, batch)
    assert torch.isfinite(loss["total"]) and loss["total"] > 0
    loss["total"].backward()


def test_mv_loss_vanishes_when_no_sample_carries_direction():
    """Back-lit batches must not push the MV head toward a coin flip."""
    outputs = {"crease": torch.zeros(2, 32, 32), "mv": torch.zeros(2, 2, 32, 32)}
    assert float(losses(outputs, fake_batch(mv_valid=0.0))["mv"]) == 0.0
    assert float(losses(outputs, fake_batch(mv_valid=1.0))["mv"]) > 0.0


def test_loss_ignores_pixels_outside_the_sheet():
    batch = fake_batch()
    outputs = {"crease": torch.zeros(2, 32, 32), "mv": torch.zeros(2, 2, 32, 32)}
    baseline = float(losses(outputs, batch)["crease"])

    masked = dict(batch)
    masked["sheet"] = batch["sheet"].clone()
    masked["sheet"][:, 20:, :] = 0.0  # blank paper turned into background
    assert float(losses(outputs, masked)["crease"]) != pytest.approx(baseline)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_counts_score_a_perfect_prediction():
    batch = fake_batch()
    outputs = {
        "crease": torch.where(batch["crease"] > 0, 10.0, -10.0),
        "mv": torch.zeros(2, 2, 32, 32),
    }
    outputs["mv"][:, 0] = 10.0  # class 0 == mountain, matching the targets

    counts = Counts()
    counts.update(outputs, batch)
    assert counts.precision == 1.0 and counts.recall == 1.0 and counts.f1 == 1.0
    assert counts.mv_accuracy == 1.0


def test_counts_report_mv_as_undefined_when_nothing_is_scorable():
    batch = fake_batch(mv_valid=0.0)
    outputs = {
        "crease": torch.where(batch["crease"] > 0, 10.0, -10.0),
        "mv": torch.zeros(2, 2, 32, 32),
    }
    counts = Counts()
    counts.update(outputs, batch)
    assert counts.f1 == 1.0
    assert np.isnan(counts.mv_accuracy)


def test_counts_penalise_over_prediction():
    batch = fake_batch()
    outputs = {"crease": torch.full((2, 32, 32), 10.0), "mv": torch.zeros(2, 2, 32, 32)}
    counts = Counts()
    counts.update(outputs, batch)
    assert counts.recall == 1.0
    assert counts.precision < 0.2  # everything called a crease
