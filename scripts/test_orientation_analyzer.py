import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

from ish_knee.preprocessing.dicom.dicom_series import DicomSeries
from ish_knee.preprocessing.dicom.dicom_slice import DicomSlice
from ish_knee.preprocessing.volume.orientation_analyzer import OrientationAnalyzer

def create_slice(
        orientation: tuple[
            float,
            float,
            float,
            float,
            float,
            float           
        ]
) -> DicomSlice:
    return DicomSlice(
        path=Path(r"D:\RSNA knee abnormality detection 2026"),
        instance_number=1,
        image_position=(0.0,0.0,0.0),
        image_orientation=orientation,
        pixel_array=np.zeros(
            (128,128),
            dtype=np.float32
        )
    )

def test_orientation(
        name: str,
        orientation: tuple[
            float,
            float,
            float,
            float,
            float,
            float           
        ],
        expected_plane: str,
) -> None:

    slice_ = create_slice(orientation)

    series = DicomSeries(
        study_instance_uid="TEST-STUDY",
        series_instance_uid="TEST-SERIES",
        slices=(slice_,)
    )

    analyzer = OrientationAnalyzer()

    result = analyzer.analyze(series)

    print()
    print(f"{name}")
    print("-" * 60)

    print("Row direction:        ", result.row_direction)
    print("Column direction:     ", result.column_direction)
    print("Slice normal:         ", result.slice_normal)
    print("Anatomical plane:     ", result.anatomical_plane)

    assert result.anatomical_plane == expected_plane

print("=" * 70)
print("ORIENTATION ANALYZER TEST")
print("=" * 70)

#sagittal
test_orientation(
    "sagittal orientation",
    (
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ),
    "sagittal",
),

# Coronal
test_orientation(
    "Coronal orientation",
    (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ),
    "coronal",
),

# Axial
test_orientation(
    "Axial orientation",
    (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ),
    "axial",
)

print()
print("=" * 70)
print("ORIENTATION ANALYZER TEST PASSED")
print("=" * 70)