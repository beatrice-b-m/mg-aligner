"""Image alignment of mammograms and flip-only realignment of images and coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, List, Optional, Tuple

from mg_aligner.metadata import (
    FovHorizontalFlip,
    FovRotation,
    Laterality,
    Manufacturer,
    PatientOrientation,
    ProjectionSide,
)

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]


class Edge(Enum):
    """An edge of the displayed image."""

    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    UNKNOWN = "unknown"


_HORIZONTAL = {Edge.LEFT, Edge.RIGHT}
_VERTICAL = {Edge.TOP, Edge.BOTTOM}


@dataclass(frozen=True)
class Alignment:
    """Where the anatomy of a mammogram sits in its pixel array.

    ``chest_wall`` is the image edge holding the chest wall. ``superior`` is the
    image edge holding the superior side of the breast; for CC-type views,
    which have no superior/inferior axis, it holds the lateral side instead.

    The reference alignment is chest wall on the left and superior (lateral for
    CC) at the top. Realignment only flips; rotated alignments are recognised
    but cannot be realigned.
    """

    chest_wall: Edge
    superior: Edge

    def __post_init__(self) -> None:
        object.__setattr__(self, "chest_wall", Edge(self.chest_wall))
        object.__setattr__(self, "superior", Edge(self.superior))
        for axis in (_HORIZONTAL, _VERTICAL):
            if self.chest_wall in axis and self.superior in axis:
                raise ValueError(f"chest_wall and superior must be on different axes: {self}")

    # Construction -----------------------------------------------------------

    @classmethod
    def reference(cls) -> "Alignment":
        """Chest wall on the left, superior (lateral for CC) at the top."""

        return cls(Edge.LEFT, Edge.TOP)

    @classmethod
    def unknown(cls) -> "Alignment":
        return cls(Edge.UNKNOWN, Edge.UNKNOWN)

    @classmethod
    def from_orientation(cls, orientation: Any, laterality: Any = None) -> "Alignment":
        """Infer alignment from DICOM Patient Orientation and image laterality.

        Each orientation value is read by its principal (first) letter: A/P
        locate the chest wall and H/F the superior side. Laterality is only
        needed when a value is plain L/R (CC-type views), to tell lateral
        from medial.
        """

        parsed = PatientOrientation.coerce(orientation)
        side = Laterality.coerce(laterality)

        edges = {}
        for axis, component in (("x", parsed.row), ("y", parsed.column)):
            anatomy = _anatomy(component, side)
            if anatomy is None:
                continue
            field, edge = _EDGE_TOWARD[axis, anatomy]
            if field in edges:
                return cls.unknown()
            edges[field] = edge
        return cls(edges.get("chest_wall", Edge.UNKNOWN), edges.get("superior", Edge.UNKNOWN))

    @classmethod
    def from_fov(
        cls,
        rotation: Any,
        horizontal_flip: Any,
        laterality: Any,
        view_position: Any,
        manufacturer: Any,
    ) -> "Alignment":
        """Infer alignment from DICOM Field of View Rotation and Horizontal Flip.

        Field of View values describe the image relative to the detector, so
        the result depends on the vendor's detector frame and on which side
        of the breast the view is taken from. Only Hologic and GE, and views
        checked against Patient Orientation, are supported; anything else
        gives an unknown alignment. Prefer :meth:`from_orientation` when
        Patient Orientation is available.
        """

        key = (
            Laterality.coerce(laterality),
            FovRotation.coerce(rotation),
            FovHorizontalFlip.coerce(horizontal_flip),
        )
        edges = _FOV_ALIGNMENT.get(key)
        vendor = Manufacturer.coerce(manufacturer)
        side = ProjectionSide.coerce(view_position)
        if not edges or vendor is Manufacturer.UNKNOWN or side is ProjectionSide.UNKNOWN:
            return cls.unknown()

        chest_wall, superior = edges
        if vendor is Manufacturer.GE:  # detector frame is rotated 180° relative to Hologic
            chest_wall, superior = _OPPOSITE[chest_wall], _OPPOSITE[superior]
        if side is ProjectionSide.REVERSE:  # detector on the other side mirrors top to bottom
            superior = _OPPOSITE[superior]
        return cls(chest_wall, superior)

    @classmethod
    def from_metadata(cls, record: Any) -> "Alignment":
        """Infer alignment from a pydicom ``Dataset``, a dict, or a pandas row.

        Uses Patient Orientation first and falls back to the Field of View
        attributes. Laterality is read from ``ImageLateralityFinal`` (EMBED),
        then ``ImageLaterality``, then ``Laterality``. Multi-frame objects such
        as DBT keep Patient Orientation and laterality per frame, so when the
        top-level attributes are missing they are read from the first
        shared or per-frame functional group.
        """

        laterality = _field(record, "ImageLateralityFinal", "ImageLaterality", "Laterality")
        if laterality is None:
            laterality = _frame_field(record, "FrameAnatomySequence", "FrameLaterality")
        orientation = _field(record, "PatientOrientation")
        if orientation is None:
            orientation = _frame_field(record, "PatientOrientationInFrameSequence", "PatientOrientation")

        alignment = cls.from_orientation(orientation, laterality)
        if alignment.is_known:
            return alignment
        return cls.from_fov(
            _field(record, "FieldOfViewRotation"),
            _field(record, "FieldOfViewHorizontalFlip"),
            laterality,
            _field(record, "ViewPosition"),
            _field(record, "Manufacturer"),
        )

    # Checks -----------------------------------------------------------------

    @property
    def is_known(self) -> bool:
        return Edge.UNKNOWN not in (self.chest_wall, self.superior)

    @property
    def is_reference(self) -> bool:
        return self == self.reference()

    # Realignment ------------------------------------------------------------

    def realign_image(self, image: Any, target: Optional["Alignment"] = None) -> Any:
        """Flip an array whose last two axes are (rows, columns) into ``target``.

        Works on 2D images and on DBT volumes shaped (frames, rows, columns).
        ``target`` defaults to the reference alignment.
        """

        flip_x, flip_y = self._flips(target)
        if flip_x:
            image = image[..., ::-1]
        if flip_y:
            image = image[..., ::-1, :]
        return image

    def realign_point(self, point: Point, *, height: int, width: int, target: Optional["Alignment"] = None) -> Point:
        """Flip a ``(y, x)`` pixel index into ``target``."""

        flip_x, flip_y = self._flips(target)
        y, x = point
        if flip_x:
            x = width - 1 - x
        if flip_y:
            y = height - 1 - y
        return y, x

    def realign_box(self, box: Box, *, height: int, width: int, target: Optional["Alignment"] = None) -> Box:
        """Flip an inclusive ``(y_min, x_min, y_max, x_max)`` pixel box into ``target``."""

        flip_x, flip_y = self._flips(target)
        y_min, x_min, y_max, x_max = box
        if flip_x:
            x_min, x_max = width - 1 - x_max, width - 1 - x_min
        if flip_y:
            y_min, y_max = height - 1 - y_max, height - 1 - y_min
        return y_min, x_min, y_max, x_max

    def realign_boxes(
        self, boxes: Iterable[Box], *, height: int, width: int, target: Optional["Alignment"] = None
    ) -> List[Box]:
        """Flip several boxes; see :meth:`realign_box`."""

        return [self.realign_box(box, height=height, width=width, target=target) for box in boxes]

    def _flips(self, target: Optional["Alignment"]) -> Tuple[bool, bool]:
        target = target or self.reference()
        for alignment in (self, target):
            if not alignment.is_known:
                raise ValueError(f"Cannot realign with an unknown alignment: {alignment}")
            if alignment.chest_wall not in _HORIZONTAL:
                raise NotImplementedError(f"Rotated alignments are not supported: {alignment}")
        return self.chest_wall is not target.chest_wall, self.superior is not target.superior


# Patient Orientation ---------------------------------------------------------

# Which image edge each anatomical direction of +x / +y points toward.
_EDGE_TOWARD = {
    ("x", "posterior"): ("chest_wall", Edge.RIGHT),
    ("x", "anterior"): ("chest_wall", Edge.LEFT),
    ("x", "superior"): ("superior", Edge.RIGHT),
    ("x", "inferior"): ("superior", Edge.LEFT),
    ("y", "posterior"): ("chest_wall", Edge.BOTTOM),
    ("y", "anterior"): ("chest_wall", Edge.TOP),
    ("y", "superior"): ("superior", Edge.BOTTOM),
    ("y", "inferior"): ("superior", Edge.TOP),
}


def _anatomy(component: str, side: Laterality) -> Optional[str]:
    """Map an orientation component to anterior/posterior/superior/inferior.

    Uses the principal (first) letter. Plain L/R counts as superior when it
    points lateral and inferior when it points medial (the CC-type convention).
    """

    principal = component[:1]
    if principal in _PRINCIPAL:
        return _PRINCIPAL[principal]
    if principal in ("L", "R") and side is not Laterality.UNKNOWN:
        return "superior" if principal == side.value else "inferior"
    return None


_PRINCIPAL = {"A": "anterior", "P": "posterior", "H": "superior", "F": "inferior"}


# Field of View ---------------------------------------------------------------

_L, _R = Laterality.LEFT, Laterality.RIGHT
_ROT0, _ROT180 = FovRotation.NONE, FovRotation.FULL
_FLIP, _NOFLIP = FovHorizontalFlip.YES, FovHorizontalFlip.NO

_OPPOSITE = {Edge.LEFT: Edge.RIGHT, Edge.RIGHT: Edge.LEFT, Edge.TOP: Edge.BOTTOM, Edge.BOTTOM: Edge.TOP}

# Alignment for Hologic standard views; from_fov adjusts it for GE and reverse views.
_FOV_ALIGNMENT = {
    (_L, _ROT180, _FLIP): (Edge.RIGHT, Edge.TOP),
    (_L, _ROT180, _NOFLIP): (Edge.LEFT, Edge.TOP),
    (_L, _ROT0, _FLIP): (Edge.LEFT, Edge.BOTTOM),
    (_L, _ROT0, _NOFLIP): (Edge.RIGHT, Edge.BOTTOM),
    (_R, _ROT180, _FLIP): (Edge.RIGHT, Edge.BOTTOM),
    (_R, _ROT180, _NOFLIP): (Edge.LEFT, Edge.BOTTOM),
    (_R, _ROT0, _FLIP): (Edge.LEFT, Edge.TOP),
    (_R, _ROT0, _NOFLIP): (Edge.RIGHT, Edge.TOP),
}


# Record access ---------------------------------------------------------------


def _field(record: Any, *names: str) -> Any:
    """Return the first present, non-missing value among ``names``."""

    for name in names:
        value = record.get(name) if hasattr(record, "get") else getattr(record, name, None)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _first_item(record: Any, sequence: str) -> Any:
    items = _field(record, sequence)
    try:
        return items[0] if items is not None and len(items) else None
    except TypeError:
        return None


def _frame_field(record: Any, macro: str, name: str) -> Any:
    """Read ``name`` from a functional group macro, shared groups first, then frame 1."""

    for groups in ("SharedFunctionalGroupsSequence", "PerFrameFunctionalGroupsSequence"):
        group = _first_item(record, groups)
        item = _first_item(group, macro) if group is not None else None
        value = _field(item, name) if item is not None else None
        if value is not None:
            return value
    return None
