
import numpy as np

from ish_knee.preprocessing.dicom.dicom_volume import DicomVolume
from ish_knee.preprocessing.volume.volume_padder import VolumePadder


def main():
    print("=" * 70)
    print("VOLUME PADDER TEST")
    print("=" * 70)

    # Create a test volume containing values from 1 through 24.
    original_array = np.arange(
        1, 25, dtype=np.float32
    ).reshape(2, 3, 4)

    volume = DicomVolume(
        study_instance_uid="TEST-STUDY",
        series_instance_uid="TEST-SERIES",
        volume=original_array,
        source_path="test",
    )

    target_shape = (4, 5, 6)

    padder = VolumePadder()
    result = padder.pad(
        volume=volume,
        target_shape=target_shape,
        constant_value=0.0,
    )

    padded = result.volume

    print("Original shape:", original_array.shape)
    print("Target shape:", target_shape)
    print("Actual shape:", padded.shape)
    print("Data type:", padded.dtype)
    print("All values finite:", np.isfinite(padded).all())

    # Verify the original voxels remain unchanged.
    np.testing.assert_array_equal(
        padded[1:3, 1:4, 1:5],
        original_array,
    )

    # Verify that the padding consists of zeros.
    assert np.all(padded[0, :, :] == 0)
    assert np.all(padded[-1, :, :] == 0)
    assert np.all(padded[:, 0, :] == 0)
    assert np.all(padded[:, -1, :] == 0)
    assert np.all(padded[:, :, 0] == 0)
    assert np.all(padded[:, :, -1] == 0)

    assert padded.shape == target_shape

    print("\nOriginal voxel values preserved: PASS")
    print("Padding values correct: PASS")
    print("Target shape correct: PASS")
    print("\nVOLUME PADDER TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
