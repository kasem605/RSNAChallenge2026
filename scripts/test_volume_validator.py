import numpy as np
from pathlib import Path

from ish_knee.preprocessing.dicom.dicom_volume import DicomVolume
from ish_knee.preprocessing.volume.volume_validator import VolumeValidator

def main():

    print("=" * 70)
    print("VOLUME VALIDATOR TEST")
    print("=" * 70)

    # ----------------------------------------------------------------------
    # Create valid synthetic volume
    # ----------------------------------------------------------------------

    volume_data = np.zeros(
        (10,128,128),
        dtype=np.float32
    )

    volume = DicomVolume(
        study_instance_uid="TEST-STUDY",
        series_instance_uid="TEST-SERIES",
        volume=volume_data,
        source_path=Path("test_series")
    )

    # ----------------------------------------------------------------------
    # Validate
    # ---------------------------------------------------------------------- 

    validator = VolumeValidator()

    validator.validate(volume)

    print()
    print("Volume validation successful")   
    print(f"Volume shape: {volume.volume}")
    print(f"Slice count:  {volume.slice_count}")
    print(f"Data type:    {volume.volume.dtype}")

    # ----------------------------------------------------------------------
    # Assertions
    # ----------------------------------------------------------------------    

    assert volume.shape == (10, 128, 128)

    assert volume.slice_count == 10

    assert volume.volume.ndim == 3

    assert np.isfinite(volume.volume).all()

    print()
    print("All assertions passed")

    print()
    print("VOLUME VALIDATOR TEST PASSED")

    print("=" * 70)

if __name__ == "__main__":
    main()
