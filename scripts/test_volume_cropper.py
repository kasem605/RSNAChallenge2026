from pathlib import Path
import numpy as np

from ish_knee.preprocessing.dicom.dicom_volume import DicomVolume
from ish_knee.preprocessing.volume.volume_cropper import VolumeCropper

def main():

    print()
    print("=" * 70)
    print("VOLUME CROPPER TEST")
    print("=" * 70)

    original_array = np.arange(
        10 * 20 * 30,
        dtype=np.float32
    ).reshape(
        10,
        20,
        30
    )

    volume = DicomVolume(
        study_instance_uid="TEST-STUDY",
        series_instance_uid="TEST-SERIES",
        volume=original_array.copy(),
        source_path=Path(r"D:\test")
    )

    cropper = VolumeCropper()

    result = cropper.crop(
        volume=volume,
        axis_0_range=(2,8),
        axis_1_range=(5,15),
        axis_2_range=(10,25)
    )

    print()
    print("Original shape:")
    print(result.shape)

    expected_shape = (
        6,
        10,
        15
    )

    print(f"Result shape: {result.shape}")

    assert result.shape == expected_shape

    print()
    print("Shape: PASS")

    assert result.study_instance_uid == "TEST-STUDY"
    assert result.series_instance_uid == "TEST-SERIES"
    assert result.source_path == Path(r"D:\test")

    print("Metadata preservation: PASS")

    assert isinstance(result.volume, np.ndarray)

    print("Numpy output: PASS")

    expected_values = original_array[
        2:8,
        5:15,
        10:25
    ]

    assert np.array_equal(
        result.volume,
        expected_values
    )

    print("Crop values: PASS")

    assert np.array_equal(
        volume.volume,
        original_array
    )

    assert result.volume.flags["C_CONTIGUOUS"]

    print("Contiguous memory: PASS")

    assert np.isfinite(
        result.volume
    ).all()

    print("Finite values: PASS")

    print()

    print("=" * 70)
    print("VOLUME CROPPER TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    main()
