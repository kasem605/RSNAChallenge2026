from pathlib import Path
import numpy as np

from ish_knee.preprocessing.dicom.dicom_volume import DicomVolume
from ish_knee.preprocessing.volume.spacing_resampler import SpacingResampler
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


    print("Original shape:", volume.volume.shape)
    print("Original spacing:", current_spacing.as_tuple)
    print("Target spacing:", target_spacing.as_tuple)

    
    # ----------------------------------------------------------------
    # Resample
    # ----------------------------------------------------------------

    resampler = SpacingResampler()

    result = resampler.resample(
        volume=volume,
        current_spacing=current_spacing,
        target_spacing=target_spacing
    )

    # ----------------------------------------------------------------
    # Display results
    # ----------------------------------------------------------------

    print()
    print("Resampled shape:")
    print(result.volume.shape)

    print()
    print("Data Type:")
    print(result.volume.dtype)

    print()
    print("All Values Finite:")
    print(np.isfinite(result.volume).all())

    print()
    print("Minimum:")
    print(result.volume.min())

    print()
    print("Maximum:")
    print(result.volume.max())

if __name__ == "__main__":
    main()