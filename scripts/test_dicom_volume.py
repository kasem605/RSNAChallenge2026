from pathlib import Path
import numpy as np

from ish_knee.preprocessing.Dicom.dicom_volume import DicomVolume

def main():

    print("=" * 70)
    print("DICOM VOLUME IMPORT TEST")
    print("=" * 70)

    # ------------------------------------------------------------------------------------------
    # Create a small test volume
    # ------------------------------------------------------------------------------------------

    test_volume = np.zeros(
        (10, 128, 128),
        dtype=np.float32
    )

    dicom_volume = DicomVolume(
        study_instance_uid="TEST-STUDY",
        series_instance_uid="TEST-SERIES",
        volume=test_volume,
        source_path=Path("D:/test")
    )

    print()
    print("Import successful")
    print(f"Study UID:      {dicom_volume.study_instance_uid}")
    print(f"Series UID:     {dicom_volume.series_instance_uid}")
    print(f"Volume Shape:   {dicom_volume.volume.shape}")
    print(f"Slice count:    {dicom_volume.slice_count}")
    print(f"Source path:    {dicom_volume.source_path}")
    print()
    print("DICOM VOLUME TEST PASSED")
    print("=" * 70)

if __name__== "__main__":
    main()

    