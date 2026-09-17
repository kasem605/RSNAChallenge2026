import numpy as np
from pathlib import Path
from ish_knee.preprocessing.Dicom.dicom_volume import DicomVolume
from ish_knee.preprocessing.volume.intensity_normalizer import IntensityNormalizer

from pathlib import Path

def main():
    print("=" * 70)
    print("INTENSITY NORMALIZER TEST")
    print("=" * 70)

    # ---------------------------------------------------------------------
    # Create synthetic volume
    # ---------------------------------------------------------------------

    volume_data = np.arange(
        1,
        101,
        dtype=np.float32
    ).reshape(
        4,
        5,
        5
    )

    volume = DicomVolume(
        study_instance_uid="TEST-STUDY",
        series_instance_uid="TEST-SERIES",
        volume=volume_data,
        source_path=Path("test_series")
    )

    print()
    print("Original volume")
    print(f"Shape: {volume.shape}")
    print(f"Mean: {volume.volume.mean():.4f}")
    print(f"Std: {volume.volume.std():.4f}")

    # ----------------------------------------------------------
    # Nomalize
    # ----------------------------------------------------------

    normalizer = IntensityNormalizer()

    normalized = normalizer.normalize(volume)

    # ----------------------------------------------------------
    # Display results
    # ----------------------------------------------------------

    print()
    print("Normalized Volume")
    print(f"Shape: {normalized.shape}")
    print(f"Mean: {normalized.volume.mean():.6f}")
    print(f"Std: {normalized.volume.std():.6f}")
    print(f"Data type: {normalized.volume.dtype}")

    # ----------------------------------------------------------
    # Validate
    # ----------------------------------------------------------

    assert normalized.shape == volume.shape

    assert normalized.volume.dtype == np.float32

    assert np.isclose(
        normalized.volume.mean(),
        0.0,
        atol=1e-6
    )

    assert np.isclose(
        normalized.volume.std(),
        1.0,
        atol=1e-6
    )

    # Original volume must remain unchanged

    assert np.array_equal(
        volume.volume,
        volume_data
    )

    print()
    print("All assertions passed")

    print()
    print("INTENSITY NORMALIZER TEST PASSED")

    print("*" * 70)

if __name__ == "__main__":
    main()