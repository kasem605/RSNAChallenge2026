from pathlib import Path

import numpy as np
from ish_knee.preprocessing.dicom.dicom_volume import DicomVolume
from ish_knee.preprocessing.volume.volume_resizer import VolumeResizer

def main():

    print()
    print("=" * 70)
    print("VOLUME RESIZER TEST")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Create synthetic MRI volume
    # ---------------------------------------------------------------

    original_volume = np.arange(
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
        volume=original_volume,
        source_path=Path("test_series")
    )

    print()
    print("Original shape:")
    print(volume.shape)

    # ---------------------------------------------------------------
    # Resize
    # ---------------------------------------------------------------   

    target_shape = (8, 16, 16)

    resizer = VolumeResizer()

    resized = resizer.resize(volume, target_shape) 

    # ---------------------------------------------------------------
    # Verify shape
    # ---------------------------------------------------------------

    print()
    print("Target shape:")
    print(target_shape)

    print()
    print("Resized shape:")
    print(resized.shape)

    assert resized.shape == target_shape

    print()
    print("Shape test: PASS")

    # ---------------------------------------------------------------
    # Verify shape
    # ---------------------------------------------------------------

    assert (resized.study_instance_uid == volume.study_instance_uid)

    assert (resized.series_instance_uid == volume.series_instance_uid)

    assert (resized.source_path == volume.source_path)

    print("Metadata preservation: PASS")

    # ---------------------------------------------------------------
    # Verify original was not modified
    # ---------------------------------------------------------------

    assert np.array_equal(volume.volume, original_volume)

    print("Original unchanged: PASS")

    # ---------------------------------------------------------------
    # Verify output is Numpy array
    # ---------------------------------------------------------------

    assert isinstance(resized.volume, np.ndarray)

    print("Numpy output: PASS")

    # ---------------------------------------------------------------
    # Verify contiguous memory
    # ---------------------------------------------------------------

    assert resized.volume.flags["C_CONTIGUOUS"]

    print("Contiguous memory: PASS")    

    # ---------------------------------------------------------------
    # Verify data type
    # ---------------------------------------------------------------

    assert np.issubdtype(resized.volume.dtype, np.floating)

    print("Floating-point output: PASS")

    print()
    print("=" * 70)
    print("VOLUME RESIZER TEST PASSED")
    print("=" * 70)
if __name__ == "__main__":
    main()