import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

from ish_knee.preprocessing.dicom.dicom_volume import DicomVolume
from ish_knee.preprocessing.volume.orientation_info import OrientationInfo
from ish_knee.preprocessing.volume.orientation_transformer import OrientationTransformer

print("=" * 70)
print("ORIENTATION TRANSFORMER TEST")
print("=" * 70)

# ------------------------------------------------------------------------
# Create synthetic volume
# ------------------------------------------------------------------------

original_volume = np.arange(
    24,
    dtype=np.float32
).reshape(
    2,
    3,
    4
)

volume = DicomVolume(
    study_instance_uid="TEST-STUDY",
    series_instance_uid="TEST-SERIES",
    volume=original_volume,
    source_path=Path(r"D:\test")
)

# ------------------------------------------------------------------------
# Create synthetic orientation information
# ------------------------------------------------------------------------

orientation = OrientationInfo(
    row_direction=(
        1.0,
        0.0,
        0.0
    ),
    column_direction=(
        0.0,
        1.0,
        0.0
    ),
    slice_normal=(
        0.0,
        0.0,
        1.0
    ), 
    anatomical_plane="axial"       
)

# ------------------------------------------------------------------------
# Transform
# ------------------------------------------------------------------------

transformer = OrientationTransformer()

result = transformer.transform( volume, orientation)

# ------------------------------------------------------------------------
# Validate
# ------------------------------------------------------------------------

print()
print("Original shape:  ", volume.shape)
print("Transformed shape:", result.shape)

print()
print("Original volume:")
print(volume.volume)

print()
print("Transformed volume:")
print(result.volume)

assert result.shape == volume.shape

assert np.array_equal(
    result.volume,
    volume.volume
)

assert result.study_instance_uid == (volume.study_instance_uid)

assert result.series_instance_uid == (volume.series_instance_uid)

assert result.source_path == volume.source_path

# Verify that the result is a copy rather than 
# the same Numpy array

assert result.volume is not volume.volume

print()
print("=" * 70)
print("ORIENTATION TRANSFORMER TEST PASSED")