"""Parsing for the DICOM and EMBED metadata values that determine alignment.

Every parser is non-throwing: malformed or missing source values become
``UNKNOWN`` so that callers can decide how to handle them.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class _CoercibleEnum(Enum):
    """Enum whose ``coerce`` maps any source value to a member or ``UNKNOWN``."""

    @classmethod
    def coerce(cls, value: Any):
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except (TypeError, ValueError):
            return cls._coerce_unknown(value)

    @classmethod
    def _coerce_unknown(cls, value: Any):
        return cls.UNKNOWN  # type: ignore[attr-defined]


def _normalized_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


class Laterality(_CoercibleEnum):
    """Breast side shown in an image."""

    LEFT = "L"
    RIGHT = "R"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _coerce_unknown(cls, value: Any) -> "Laterality":
        aliases = {"L": cls.LEFT, "LEFT": cls.LEFT, "R": cls.RIGHT, "RIGHT": cls.RIGHT}
        return aliases.get(_normalized_text(value), cls.UNKNOWN)


@dataclass(frozen=True)
class PatientOrientation:
    """DICOM Patient Orientation (0020,0020): the patient directions of the image axes.

    ``row`` is the direction of increasing column index (+x) and ``column`` is
    the direction of increasing row index (+y). Each is a code such as ``"P"``
    or ``"FL"`` built from A, P, R, L, H, F, with the principal direction first
    (DICOM PS3.3 C.7.6.1.1.1). Invalid components are stored as ``""``.
    """

    row: str
    column: str

    @classmethod
    def coerce(cls, value: Any) -> "PatientOrientation":
        """Parse a pydicom value, ``"P\\L"``, ``"['P', 'L']"``, or a pair."""

        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            text = value.strip()
            if text.startswith(("[", "(")):
                try:
                    value = ast.literal_eval(text)
                except (SyntaxError, ValueError):
                    value = None
            else:
                value = text.split("\\")
        try:
            row, column = value
        except (TypeError, ValueError):
            row = column = None
        return cls(_component(row), _component(column))


def _component(value: Any) -> str:
    text = _normalized_text(value)
    return text if re.fullmatch(r"[APRLHF]{1,3}", text) else ""


class FovRotation(_CoercibleEnum):
    """DICOM Field of View Rotation (0018,7032), in degrees."""

    NONE = 0.0
    FULL = 180.0
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _coerce_unknown(cls, value: Any) -> "FovRotation":
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return cls.UNKNOWN
        return {0.0: cls.NONE, 180.0: cls.FULL}.get(numeric, cls.UNKNOWN)


class FovHorizontalFlip(_CoercibleEnum):
    """DICOM Field of View Horizontal Flip (0018,7034)."""

    YES = "YES"
    NO = "NO"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _coerce_unknown(cls, value: Any) -> "FovHorizontalFlip":
        aliases = {
            "Y": cls.YES, "YES": cls.YES, "TRUE": cls.YES, "1": cls.YES,
            "N": cls.NO, "NO": cls.NO, "FALSE": cls.NO, "0": cls.NO,
        }
        return aliases.get(_normalized_text(value), cls.UNKNOWN)
