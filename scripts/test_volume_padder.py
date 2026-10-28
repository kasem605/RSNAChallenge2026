from pathlib import Path
import numpy as np

from ish_knee.preprocessing.dicom.dicom_volume import DicomVolume
from ish_knee.preprocessing.volume.volume_padder import VolumePadder

def main():

    print()
    print("=" * 70)
    print("VOLUME PADDER TEST")
    print("=" * 70)

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

    padder = VolumePadder()

    result = padder.pad(
        volume=volume,
        target_shape=(8, 10, 12),
        constant_value=0.0
    )

    print()
    print("Original shape:")
    print(volume.shape)
    print()
    print("Padded shape:")
    print(result.shape)

    expected_shape = (
        8,
        10,
        12
    )

    assert result.shape == expected_shape

    print()
    print("Shape: PASS")

    assert result.study_instance_uid == "TEST-STUDY"
    assert result.series_instance_uid == "TEST-SERIES"
    assert result.source_path == Path(r"D:\test")

    print("Metadata preservation: PASS")

    assert isinstance(
        result.volume,
        np.ndarray
    )

    print("Numpy output: PASS")

    assert np.array_equal(
        result.volume[2:6, 2:8, 2:10],
        original_array
    )

    print("Original data preserved: PASS")

    assert np.all(result.volume[:2,:,:] == 0)

    assert np.all(result.volume[6:,:,:] == 0)

    assert np.all(result.volume[:,:2,:] == 0)

    assert np.all(result.volume[:,8:,:] == 0)

    assert np.all(result.volume[:,:,:2] == 0)

    assert np.all(result.volume[:,:,10:] == 0)

    print("Zero padding: PASS")

    assert np.array_equal(
        volume.volume,
        original_array
    )

    print("Original volume unchanged: PASS")

    assert result.volume.flags["C_CONTIGUOUS"]

    print("Contiguoud memory: PASS")

    assert np.isfinite(
        result.volume
    ).all()

    print("Finite values: PASS")

    print()
    print("=" * 70)
    print("VOLUME PADDER TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    main()

