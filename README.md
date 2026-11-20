# mg-aligner

A minimal Python library for checking the orientation of mammography images and
realigning them, and the coordinates on them, into a common frame.

Mammograms from different vendors and pipelines are stored with different
on-screen orientations: the chest wall may sit on the left or the right, and the
superior edge may be at the top or the bottom. `mg-aligner` works out an image's
orientation from its metadata and flips the image, points, or bounding boxes into
a single reference frame (or any other frame you choose).

It is built primarily for the [Emory Breast Imaging Dataset
(EMBED)](https://github.com/Emory-HITI/EMBED_Open_Data), and aims to support
synthetic 2D, FFDM, and DBT images from multiple manufacturers.

> **Status:** early development; the API may still change.

## Installation

```bash
uv add git+https://github.com/beatrice-b-m/mg-aligner.git
```

or with pip:

```bash
pip install git+https://github.com/beatrice-b-m/mg-aligner.git
```

The library has no runtime dependencies. Images are expected to be NumPy-style
arrays.

## Usage

An `Alignment` records which image edge holds the **chest wall** and which holds
the **superior** side of the breast (the lateral side for CC views). The
reference alignment is chest wall on the left, superior/lateral at the top.

```python
import pydicom
from mg_aligner import Alignment

ds = pydicom.dcmread("image.dcm")
alignment = Alignment.from_metadata(ds)   # also accepts a dict or pandas row

alignment.is_known       # False if the metadata does not determine it
alignment.is_reference   # True if no flip is needed

image = alignment.realign_image(ds.pixel_array)
box = alignment.realign_box(roi, height=ds.Rows, width=ds.Columns)
```

### Constructors

| Method | Uses |
| --- | --- |
| `Alignment.from_metadata(record)` | Patient Orientation, falling back to Field of View attributes. Reads `ImageLateralityFinal` (EMBED), `ImageLaterality`, or `Laterality`. |
| `Alignment.from_orientation(orientation, laterality, view)` | DICOM Patient Orientation, e.g. `["P", "L"]`, `"P\\L"`, or `"['P', 'L']"`. |
| `Alignment.from_fov(rotation, horizontal_flip, laterality)` | DICOM Field of View Rotation (0 or 180) and Horizontal Flip. |
| `Alignment(chest_wall, superior)` | Explicit edges, e.g. `Alignment("right", "top")`. |

Missing or malformed metadata gives an unknown alignment rather than an error.

### Realignment

`realign_image`, `realign_point`, `realign_box`, and `realign_boxes` flip into
the reference alignment, or into any `target=` alignment, so the same calls map
results back into the original frame:

```python
original = Alignment.reference().realign_box(box, height=h, width=w, target=alignment)
```

- Images: the last two axes are (rows, columns), so 2D images and DBT volumes
  shaped (frames, rows, columns) both work. Frame indices are unchanged.
- Coordinates are inclusive pixel indices: points are `(y, x)` and boxes are
  `(y_min, x_min, y_max, x_max)`, matching EMBED `ROI_coords`.
- Only flips are supported. Realigning an unknown alignment raises `ValueError`;
  a rotated one (chest wall at the top or bottom) raises `NotImplementedError`.

See [`examples/realign_synthetic.py`](examples/realign_synthetic.py) for a
runnable walk-through.

## Validation

[`validation/embed_metadata.py`](validation/embed_metadata.py) checks the two
metadata routes against each other on an EMBED image-metadata CSV
(`metadata_all_cohorts_v1c`). For each 2D, 3D, and C-View image, it infers the
alignment from Patient Orientation and from the Field of View attributes, then
reports agreement by manufacturer, model, image type, and view. It also lists
the input values behind any disagreement or unknown result. Output is
aggregate counts only.

```bash
uv run --group validation python validation/embed_metadata.py \
    path/to/metadata_all_cohorts_v1c.csv --out validation_results/
```

## Development

The project uses [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```
