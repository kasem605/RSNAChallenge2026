from pathlib import Path
import numpy as np

from ish_knee.preprocessing.dicom.dicom_volume import DicomVolume
from ish_knee.preprocessing.volume.spacing_resampler import ScacingResampler
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing

def main():

    print()
    print("=" * 70)
    print("SPACING RESAMPLER TEST")
    print("=" * 70)

    # ----------------------------------------------------------------
    # CReate a small synthetic 3-D volume
    # ----------------------------------------------------------------

    original_array = np.arange(
        4 * 6 * 8,
        dtype=np.float32
    ).reshape(
        4,
        6,
        8
    )

    volume = DicomVolume(
        study_instance_uid="TEST-STUDY",
        series_instance_uid="TEST-SERIES",
        volume=original_array.copy(),
        source_path=Path(r"D:\test")
    )

    # ----------------------------------------------------------------
    # Define current physical spacing
    # ----------------------------------------------------------------

    current_spacing = VoxelSpacing(
        spacing_axis_0=2.0,
        spacing_axis_1=2.0,
        spacing_axis_2=2.0    
    )

    # ----------------------------------------------------------------
    # Define target physical spacing
    # ----------------------------------------------------------------

    target_spacing = VoxelSpacing(
        spacing_axis_0=1.0,
        spacing_axis_1=1.0,
        spacing_axis_2=1.0    
    )

    # ----------------------------------------------------------------
    # Resample
    # ----------------------------------------------------------------

    resampler = ScacingResampler()

    result = resampler.resample(
        volume=volume,
        current_spacing=current_spacing,
        target_spacing=target_spacing
    )

    # ----------------------------------------------------------------
    # Display results
    # ----------------------------------------------------------------

    print()
    print("Original shape:")
    print(volume.shape)

    print()
    print("Resampled shape:")
    print(result.shape)

    print()
    print("Original spacing:")
    print(current_spacing.as_tuple)

    print()
    print("Target spacing:")
    print(target_spacing.as_tuple)

    # ----------------------------------------------------------------
    # Validate shape
    # ----------------------------------------------------------------

    expected_shape = (
        8,
        12,
        16
    )

    assert result.shape == expected_shape

    print()
    print("Shape:   PASS")

    # ----------------------------------------------------------------
    # Validate metadata
    # ----------------------------------------------------------------

    assert result.study_instance_uid == "TEST-STUDY"

    assert result.series_instance_uid == "TEST-SERIES"

    assert result.source_path == Path(r"D:\test")

    print("Metadata preservation:   PASS")

    # ----------------------------------------------------------------
    # Validate output type
    # ----------------------------------------------------------------

    assert isinstance(result.volume, np.ndarray)

    print("Numpy output:    PASS")

    # ----------------------------------------------------------------
    # Validate original volume was not changed
    # ----------------------------------------------------------------

    assert np.array_equal(volume.volume, original_array)

    print("Original volume unchanged:    PASS")

    # ----------------------------------------------------------------
    # Validate contiguous memory
    # ----------------------------------------------------------------

    assert result.volume.flags["C_CONTIGUOUS"]

    print("Contiguous memory:    PASS")

    # ----------------------------------------------------------------
    # Validate output contains finite values
    # ----------------------------------------------------------------

    assert np.isfinite(result.volume).all()

    print("Finite values:   PASS")

    print()
    print("=" * 70)
    print("SPACING RESAMPLER TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    main()