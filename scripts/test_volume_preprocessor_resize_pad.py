import numpy as np
from pathlib import Path
from ish_knee.preprocessing.dicom.dicom_volume import DicomVolume
from ish_knee.preprocessing.volume.preprocessing_config import PreprocessingConfig
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
from ish_knee.preprocessing.volume.volume_preprocessor import VolumePreprocessor

def main():

    print()
    print("=" * 70)
    print("VOLUME PREPROCESSOR RESIZE + PAD TEST")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Create synthetic MRI volume
    # ---------------------------------------------------------------

    original_array = np.ones(
        (6, 8, 10),
        dtype=np.float32
    )

    volume = DicomVolume(
        study_instance_uid="TEST-STUDY",
        series_instance_uid="TEST-SERIES",
        volume=original_array.copy(),
        source_path=Path(r"D:\test")
    )

    print()
    print("Original shape:")
    print(volume.shape)

    # ---------------------------------------------------------------
    # Current voxel spacing
    # ---------------------------------------------------------------    
    
    current_spacing = VoxelSpacing(
        spacing_axis_0=1.0,
        spacing_axis_1=1.0,
        spacing_axis_2=1.0           
    )

    # ---------------------------------------------------------------
    # Target configuration
    #
    # Current: (6, 8, 10)
    # Traget:  (8, 6, 12)
    #
    # Expected:
    #
    # axis 0: 6 -> 6, then pad to 8
    # axis 1: 8 -> 6, resize down
    # axis 2: 10 -> 10, then pad to 12
    # ---------------------------------------------------------------

    config = PreprocessingConfig(
        target_spacing=VoxelSpacing(
            spacing_axis_0=1.0,
            spacing_axis_1=1.0,
            spacing_axis_2=1.0
        ),
        target_shape=(8, 6, 12),
        normalize_intensity=True,
        padding_value=0.0
    )

    preprocessor = VolumePreprocessor(config=config)

    # ----------------------------------------------------------------
    # Process
    # ----------------------------------------------------------------

    
    result = preprocessor.preprocess(
        volume=volume,
        current_spacing=current_spacing
    )

    print()
    print("Original shape:")
    print(volume.volume.shape)

    print()
    print("Target shape:")
    print(config.target_shape)

    print()
    print("Result shape:")
    print(result.volume.shape)

    # ----------------------------------------------------------------
    # Verify final shape
    # ----------------------------------------------------------------

    assert result.volume.shape == (
        8,
        6,
        12
    )

    print("Final shape: PASS")

    # ----------------------------------------------------------------
    # Verify metadata
    # ----------------------------------------------------------------

    assert result.study_instance_uid == "TEST-STUDY"
    assert result.series_instance_uid == "TEST-SERIES"
    assert result.source_path == Path(r"D:\test")

    print("Metadata preservation: PASS")

    # ----------------------------------------------------------------
    # Verify output
    # ----------------------------------------------------------------

    assert isinstance(result.volume, np.ndarray)

    print("Numpy output: PASS")

    # ----------------------------------------------------------------
    # Verify original unchanged
    # ----------------------------------------------------------------    

    assert np.array_equal(
        volume.volume,
        original_array
    )

    print("Original volume unchanged: PASS")

    # ----------------------------------------------------------------
    # Verify contiguous memory
    # ----------------------------------------------------------------

    assert result.volume.flags["C_CONTIGUOUS"]

    print("Contiguous memory: PASS")

    # ----------------------------------------------------------------
    # Verify finite values
    # ----------------------------------------------------------------

    assert np.isfinite(result.volume).all()

    print("Finite values: PASS")

    print()
    print("=" * 70)
    print("VOLUME PREPROCESSOR TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    main()
