
import numpy as np

from ish_knee.preprocessing.dicom.dicom_volume import DicomVolume
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
from ish_knee.preprocessing.volume.preprocessing_config import PreprocessingConfig
from ish_knee.preprocessing.volume.volume_preprocessor import VolumePreprocessor


def main():
    print("=" * 70)
    print("VOLUME PREPROCESSOR TEST")
    print("=" * 70)

    # Create a synthetic 3-D MRI volume.
    original_array = np.arange(
        1, 33, dtype=np.float32
    ).reshape(2, 4, 4)

    volume = DicomVolume(
        study_instance_uid="TEST-STUDY",
        series_instance_uid="TEST-SERIES",
        volume=original_array,
        source_path="test",
    )

    current_spacing = VoxelSpacing(
        spacing_axis_0=2.0,
        spacing_axis_1=2.0,
        spacing_axis_2=2.0,
    )

    config = PreprocessingConfig(
        target_spacing=VoxelSpacing(
            spacing_axis_0=1.0,
            spacing_axis_1=1.0,
            spacing_axis_2=1.0,
        ),
        target_shape=(128, 128, 128),
        normalize_intensity=True,
        padding_value=0.0,
    )

    print("Original shape:", volume.volume.shape)
    print("Original spacing:", current_spacing.as_tuple)
    print("Target spacing:", config.target_spacing.as_tuple)
    print("Target shape:", config.target_shape)

    preprocessor = VolumePreprocessor(config)

    result = preprocessor.preprocess(
        volume=volume,
        current_spacing=current_spacing,
    )

    output = result.volume

    print("\nFinal shape:", output.shape)
    print("Data type:", output.dtype)
    print("All values finite:", np.isfinite(output).all())
    print("Minimum intensity:", output.min())
    print("Maximum intensity:", output.max())
    print("Contiguous array:", output.flags.c_contiguous)

    # Validate the final result.
    assert output.shape == (128, 128, 128)
    assert np.isfinite(output).all()
    assert output.min() >= 0.0
    assert output.max() <= 1.0
    assert output.dtype == np.float32

    print("\nFinal shape: PASS")
    print("Finite values: PASS")
    print("Intensity range: PASS")
    print("Data type: PASS")
    print("\nVOLUME PREPROCESSOR TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
