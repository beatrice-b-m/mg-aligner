"""Compare Patient Orientation and Field of View alignment on EMBED image metadata.

For every mammography image row, infer the alignment twice, once from
PatientOrientation and once from the Field of View attributes, and report how
often they agree, broken down by manufacturer, model, image type and view.

The output contains only aggregate counts and metadata values; no patient,
exam, or image identifiers are read.

Usage:
    uv run --group validation python validation/embed_metadata.py \
        path/to/metadata_all_cohorts_v1c.csv [--out results/] [--all-image-types]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mg_aligner import Alignment, Edge

INPUTS = [
    "PatientOrientation",
    "ImageLateralityFinal",
    "ViewPosition",
    "FieldOfViewRotation",
    "FieldOfViewHorizontalFlip",
]
GROUPS = ["Manufacturer", "ManufacturerModelName", "FinalImageType"]
MAMMOGRAPHY_TYPES = {"2D", "3D", "cview"}  # excludes ROI_SS/ROI_SSC captures and "other"


def status(alignment: Alignment) -> str:
    if not alignment.is_known:
        return "unknown"
    if alignment.chest_wall not in (Edge.LEFT, Edge.RIGHT):
        return "rotated"
    return "known"


def compare(orientation, laterality, view, rotation, flip) -> dict:
    by_orientation = Alignment.from_orientation(orientation, laterality)
    by_fov = Alignment.from_fov(rotation, flip, laterality)
    o, f = status(by_orientation), status(by_fov)
    if o == "known" and f == "known":
        if by_orientation == by_fov:
            outcome = "agree"
        elif by_orientation.chest_wall is by_fov.chest_wall:
            outcome = "disagree: superior only"
        elif by_orientation.superior is by_fov.superior:
            outcome = "disagree: chest wall only"
        else:
            outcome = "disagree: both"
    else:
        outcome = f"orientation {o} / fov {f}"
    return {
        "outcome": outcome,
        "orientation_alignment": f"{by_orientation.chest_wall.value}/{by_orientation.superior.value}",
        "fov_alignment": f"{by_fov.chest_wall.value}/{by_fov.superior.value}",
        "is_reference": by_orientation.is_reference if o == "known" else by_fov.is_reference,
    }


def load(path: Path, all_image_types: bool) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=INPUTS + GROUPS, dtype=str, low_memory=False)
    if not all_image_types:
        frame = frame[frame["FinalImageType"].isin(MAMMOGRAPHY_TYPES)]
    frame["ViewPosition"] = frame["ViewPosition"].str.strip().str.upper()
    return frame


def summarize(frame: pd.DataFrame) -> dict:
    # Each distinct combination of inputs is evaluated once, then weighted by its row count.
    combos = frame.groupby(INPUTS + GROUPS, dropna=False).size().rename("images").reset_index()
    results = pd.DataFrame(
        [compare(*row) for row in combos[INPUTS].itertuples(index=False, name=None)],
        index=combos.index,
    )
    combos = pd.concat([combos, results], axis=1)

    def table(by):
        counts = combos.groupby(by + ["outcome"], dropna=False)["images"].sum().unstack(fill_value=0)
        counts.insert(0, "total", counts.sum(axis=1))
        return counts.sort_values("total", ascending=False)

    not_agreeing = combos[combos["outcome"] != "agree"]
    return {
        "overall": combos.groupby("outcome")["images"].sum().sort_values(ascending=False).to_frame(),
        "by_manufacturer": table(["Manufacturer"]),
        "by_model": table(["Manufacturer", "ManufacturerModelName"]),
        "by_image_type_and_view": table(["FinalImageType", "ViewPosition"]),
        "reference_share_by_manufacturer": combos[combos["outcome"] == "agree"]
        .groupby(["Manufacturer", "FinalImageType", "is_reference"], dropna=False)["images"].sum()
        .unstack(fill_value=0),
        "non_agreeing_inputs": not_agreeing.groupby(
            ["outcome", "Manufacturer"] + INPUTS + ["orientation_alignment", "fov_alignment"], dropna=False
        )["images"].sum().sort_values(ascending=False).to_frame(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metadata_csv", type=Path)
    parser.add_argument("--out", type=Path, help="also write each table as CSV into this directory")
    parser.add_argument("--all-image-types", action="store_true", help="include ROI captures and 'other' rows")
    parser.add_argument("--top", type=int, default=40, help="rows to print from the non-agreeing-inputs table")
    args = parser.parse_args()

    frame = load(args.metadata_csv, args.all_image_types)
    print(f"{len(frame):,} image rows evaluated\n")
    tables = summarize(frame)

    with pd.option_context("display.width", 200, "display.max_columns", None, "display.max_rows", None):
        for name, result in tables.items():
            shown = result.head(args.top) if name == "non_agreeing_inputs" else result
            print(f"## {name}\n{shown.to_string()}\n")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, result in tables.items():
            result.to_csv(args.out / f"{name}.csv")
        print(f"Tables written to {args.out}")


if __name__ == "__main__":
    main()
