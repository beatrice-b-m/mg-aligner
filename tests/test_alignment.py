import itertools

import numpy as np
import pytest

from mg_aligner import Alignment, Edge

FLIPPABLE = [Alignment(c, s) for c, s in itertools.product([Edge.LEFT, Edge.RIGHT], [Edge.TOP, Edge.BOTTOM])]


@pytest.mark.parametrize("source", FLIPPABLE)
@pytest.mark.parametrize("target", FLIPPABLE)
def test_coordinates_follow_the_pixels_they_mark(source, target):
    image = np.arange(4 * 5).reshape(4, 5)
    box = (1, 0, 2, 3)  # inclusive (y_min, x_min, y_max, x_max)
    marked = image[box[0] : box[2] + 1, box[1] : box[3] + 1]

    flipped = source.realign_image(image, target)
    y0, x0, y1, x1 = source.realign_box(box, height=4, width=5, target=target)
    assert sorted(flipped[y0 : y1 + 1, x0 : x1 + 1].ravel()) == sorted(marked.ravel())

    y, x = source.realign_point((1, 0), height=4, width=5, target=target)
    assert flipped[y, x] == image[1, 0]


@pytest.mark.parametrize("source", FLIPPABLE)
def test_round_trip_restores_the_original(source):
    image = np.arange(3 * 4 * 5).reshape(3, 4, 5)  # DBT-shaped: (frames, rows, columns)
    reference = Alignment.reference()
    restored = reference.realign_image(source.realign_image(image), target=source)
    assert np.array_equal(restored, image)

    box = (0, 0, 3, 4)
    there = source.realign_box(box, height=4, width=5)
    assert reference.realign_box(there, height=4, width=5, target=source) == box


def test_realigning_unknown_raises():
    with pytest.raises(ValueError):
        Alignment(Edge.LEFT, Edge.UNKNOWN).realign_image(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        Alignment.reference().realign_image(np.zeros((2, 2)), target=Alignment.unknown())


def test_realigning_rotated_raises():
    with pytest.raises(NotImplementedError):
        Alignment(Edge.TOP, Edge.LEFT).realign_image(np.zeros((2, 2)))


def test_edges_must_be_on_different_axes():
    with pytest.raises(ValueError):
        Alignment(Edge.LEFT, Edge.RIGHT)


@pytest.mark.parametrize(
    "orientation, laterality",
    [
        (["A", "L"], "R"),  # CC: +y points medial, so lateral is at the top
        ("A\\FL", "R"),  # MLO: principal letter F is inferior
        ("['A', 'F']", None),  # H/F needs no laterality
    ],
)
def test_orientation_reference_cases(orientation, laterality):
    assert Alignment.from_orientation(orientation, laterality).is_reference


def test_orientation_mirror_case():
    # Left CC displayed chest-wall-right: +x posterior, +y medial.
    assert Alignment.from_orientation(["P", "R"], "L") == Alignment(Edge.RIGHT, Edge.TOP)


def test_head_pointing_down_puts_superior_at_the_bottom():
    assert Alignment.from_orientation(["P", "H"], "R") == Alignment(Edge.RIGHT, Edge.BOTTOM)


def test_transposed_orientation_is_recognised_as_rotated():
    assert Alignment.from_orientation(["L", "A"], "R") == Alignment(Edge.TOP, Edge.LEFT)


@pytest.mark.parametrize(
    "orientation, laterality",
    [
        (None, "L"),
        ("garbage", "L"),
        (["P", "X"], "L"),
        (["P", "R"], None),  # plain L/R needs laterality
        (["A", "P"], "L"),  # both axes claim the chest wall
    ],
)
def test_orientation_incomplete_or_malformed_is_not_known(orientation, laterality):
    assert not Alignment.from_orientation(orientation, laterality).is_known


def test_fov_unsupported_rotation_is_unknown():
    assert not Alignment.from_fov(90, "NO", "L", "CC", "HOLOGIC, Inc.").is_known


@pytest.mark.parametrize("view, manufacturer", [("CC", "iCAD, Inc."), ("CC", None), ("SPOT", "HOLOGIC, Inc."), (None, "HOLOGIC, Inc.")])
def test_fov_unvalidated_vendor_or_view_is_unknown(view, manufacturer):
    assert not Alignment.from_fov(0, "NO", "L", view, manufacturer).is_known


def test_fov_flip_moves_only_the_chest_wall():
    plain = Alignment.from_fov(0, "NO", "R", "CC", "HOLOGIC, Inc.")
    flipped = Alignment.from_fov(0, "YES", "R", "CC", "HOLOGIC, Inc.")
    assert plain.superior is flipped.superior
    assert plain.chest_wall is not flipped.chest_wall


def test_fov_ge_is_hologic_rotated_180():
    hologic = Alignment.from_fov(0, "NO", "L", "CC", "HOLOGIC, Inc.")
    ge = Alignment.from_fov(0, "NO", "L", "CC", "GE MEDICAL SYSTEMS")
    assert hologic.chest_wall is not ge.chest_wall
    assert hologic.superior is not ge.superior


def test_fov_reverse_view_mirrors_only_the_superior_edge():
    standard = Alignment.from_fov(0, "NO", "L", "ML", "HOLOGIC, Inc.")
    reverse = Alignment.from_fov(0, "NO", "L", "LM", "HOLOGIC, Inc.")
    assert standard.chest_wall is reverse.chest_wall
    assert standard.superior is not reverse.superior


# (PatientOrientation, laterality, view, FOV rotation, FOV flip, manufacturer) seen in EMBED metadata.
@pytest.mark.parametrize(
    "orientation, laterality, view, rotation, flip, manufacturer",
    [
        (["A", "R"], "L", "CC", 0.0, "NO", "GE MEDICAL SYSTEMS"),
        (["P", "FL"], "R", "MLO", 180.0, "NO", "GE HEALTHCARE"),
        (["P", "F"], "L", "LM", 0.0, "NO", "HOLOGIC, Inc."),
        (["P", "H"], "R", "LM", 0.0, "NO", "HOLOGIC, Inc."),
        (["P", "F"], "L", "LM", 180.0, "NO", "GE MEDICAL SYSTEMS"),
        (["P", "F"], "R", "MLO", 0.0, "NO", "HOLOGIC, Inc."),
        (["A", "L"], "R", "CV", 0.0, "YES", "HOLOGIC, Inc."),
    ],
)
def test_fov_matches_orientation_on_observed_metadata(orientation, laterality, view, rotation, flip, manufacturer):
    by_orientation = Alignment.from_orientation(orientation, laterality)
    assert by_orientation.is_known
    assert Alignment.from_fov(rotation, flip, laterality, view, manufacturer) == by_orientation


def test_from_metadata_prefers_orientation_then_falls_back_to_fov():
    record = {
        "ImageLateralityFinal": "R",
        "ViewPosition": "CC ",
        "Manufacturer": "HOLOGIC, Inc.",
        "PatientOrientation": "['A', 'L']",
        "FieldOfViewRotation": 0.0,
        "FieldOfViewHorizontalFlip": "YES",
    }
    assert Alignment.from_metadata(record).is_reference

    record["PatientOrientation"] = float("nan")
    assert Alignment.from_metadata(record) == Alignment.from_fov(0.0, "YES", "R", "CC", "HOLOGIC, Inc.")


def test_from_metadata_reads_attributes():
    class Dataset:  # stands in for pydicom.Dataset attribute access
        ImageLaterality = "L"
        ViewPosition = "MLO"
        PatientOrientation = ["P", "FR"]

    assert Alignment.from_metadata(Dataset()) == Alignment(Edge.RIGHT, Edge.TOP)


def test_from_metadata_with_nothing_is_unknown():
    assert not Alignment.from_metadata({}).is_known
