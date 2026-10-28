from pathlib import Path
import numpy as np

from ish_knee.preprocessing.dicom.dicom_volume import DicomVolume
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
from ish_knee.preprocessing.volume.preprocessing_config import PreprocessingConfig
from ish_knee.preprocessing.volume.volume_preprocessor import VolumePreprocessor

def main():

    print()
    print("=" * 70)
    print("VOLUME PREPROCESSOR TEST")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Create synthetic MRI volume
    # ---------------------------------------------------------------

    original_array = np.arange(
        4 * 8 * 8,
        dtype=np.float32
    ).reshape(
        4,
        8,
        8
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
    # Define current voxel spacing
    # ---------------------------------------------------------------   

    current_spacing = VoxelSpacing(
        spacing_axis_0=2.0,
        spacing_axis_1=2.0,
        spacing_axis_2=2.0
    ) 

    # ---------------------------------------------------------------
    # Define preprocessing configuration
    # ---------------------------------------------------------------

    config = PreprocessingConfig(
        target_spacing=VoxelSpacing(
            spacing_axis_0=1.0,
            spacing_axis_1=1.0,
            spacing_axis_2=1.0
        ),
        target_shape=(8, 12, 16),
        normal_intensity=True,
        padding_value=0.0
    )

    # ---------------------------------------------------------------
    # Create preprocessor
    # ---------------------------------------------------------------

    prepprocessor = VolumePreprocessor(config=config)

    # ---------------------------------------------------------------
    # Run preprocessing
    # ---------------------------------------------------------------  

    result = prepprocessor.preprocess(
        volume=volume,
        current_spacing=current_spacing
    )

    print()
    print("Original shape:")
    print(volume.volume.shape)

    print()
    print("Processed shape:")
    print(result.volume.shape)

    # ---------------------------------------------------------------
    # Verify final shape
    # ---------------------------------------------------------------  

    assert result.volume.shape == (8, 12, 16)   

    print("Final shape: PASS")

    # ---------------------------------------------------------------
    # Verify metadata
    # --------------------------------------------------------------- 

    assert result.study_instance_uid == ("TEST-STUDY")

    assert result.series_instance_uid == ("TEST-SERIES")

    assert result.source_path == Path(r"D:\test")

    print("Metadata preservation: PASS")
    
    # ---------------------------------------------------------------
    # Verify numpy output
    # --------------------------------------------------------------- 

    assert isinstance(result.volume, np.ndarray)

    print("Numpy output: PASS")

    # ---------------------------------------------------------------
    # Verify original volume unchanged
    # --------------------------------------------------------------- 

    assert np.array_equal(volume.volume, original_array)

    print("Original volume unchanged: PASS")

    # ---------------------------------------------------------------
    # Verify contiguous memory
    # ---------------------------------------------------------------     

    assert result.volume.flags["C_CONTIGUOUS"]

    print("Contiguous memory: PASS")

    # ---------------------------------------------------------------
    # Verify finite values
    # --------------------------------------------------------------- 

    assert np.isfinite(result.volume).all

    print("Finite value: PASS")

    print()
    print("=" * 70)
    print("VOLUME PREPROCESSOR TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    main()