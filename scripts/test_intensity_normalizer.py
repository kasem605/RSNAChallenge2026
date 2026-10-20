import numpy as np
from pathlib import Path
from ish_knee.preprocessing.dicom.dicom_volume import DicomVolume
from ish_knee.preprocessing.volume.intensity_normalizer import IntensityNormalizer

from pathlib import Path

def main():

    print("=" * 70)
    print("INTENSITY NORMALIZER TEST")
    print("=" * 70)

    # ---------------------------------------------------------------------
    # Create synthetic volume
    # ---------------------------------------------------------------------

    original_array = np.array(
        [
            [
                [10, 20, 30],
                [40, 50, 60]
            ],
            [
                [70, 80, 90],
                [100, 110, 120]
            ],
        ],
        dtype=np.int16
    )

    volume = DicomVolume(
        study_instance_uid="TEST-STUDY",
        series_instance_uid="TEST-SERIES",
        volume= original_array.copy(),
        source_path=Path(r"D:\test")
    )

    # ----------------------------------------------------------
    # Nomalize
    # ----------------------------------------------------------

    normalizer = IntensityNormalizer()

    result = normalizer.normalize(volume)

    print()
    print("Original shape:")
    print(volume.volume.shape)

    print()
    print("Normalized shape:")
    print(result.volume.shape)
    
    # ----------------------------------------------------------
    # verify shape
    # ----------------------------------------------------------

    assert result.volume.shape == (
        2,
        2,
        3
    )

    print("Shape: PASS")

    # ----------------------------------------------------------
    # verify data type
    # ----------------------------------------------------------

    assert result.volume.dtype == np.float32

    print("Float32 output: PASS")

    # ----------------------------------------------------------
    # verify range
    # ----------------------------------------------------------   

    assert np.isclose(
        result.volume.min(),
        0.0
    )

    assert np.isclose(
        result.volume.max(),
        1.0
    )

    print("Normalized range [0, 1]: PASS")

    # ----------------------------------------------------------
    # verify known volumes
    # ---------------------------------------------------------- 

    assert np.isclose(result.volume[0, 0, 0], 0.0)

    assert np.isclose(result.volume[1, 1, 2], 1.0)  

    assert np.isclose(result.volume[0, 1, 1], 40.0 / 110.0) 

    print("Normalized values: PASS")

    # ----------------------------------------------------------
    # verify metadata preservation
    # ---------------------------------------------------------- 

    assert result.study_instance_uid == "TEST-STUDY"

    assert result.series_instance_uid == "TEST-SERIES"

    assert result.source_path == Path(r"D:\test")

    print("Metadata preservation: PASS")

    # ----------------------------------------------------------
    # verify original volume unchanged
    # ----------------------------------------------------------  

    assert np.array_equal(volume.volume, original_array)

    print("Original volume unchanged: PASS")

    # ----------------------------------------------------------
    # verify numpy output
    # ---------------------------------------------------------- 

    assert isinstance(result.volume, np.ndarray)

    print("Numpy output: PASS")

    # ----------------------------------------------------------
    # verify contiguous memory
    # ----------------------------------------------------------   

    assert result.volume.flags["C_CONTIGUOUS"]

    print("Contiguous memory: PASS")

    # ----------------------------------------------------------
    # verify finite values
    # ----------------------------------------------------------  

    assert np.isfinite(result.volume).all()

    print("Finite values: PASS")

    # ----------------------------------------------------------
    # Test constant-intensity volume
    # ---------------------------------------------------------- 
    
    constant_array = np.full(
        (2, 3, 4),
        25.0,
        dtype=np.float32
    )

    constant_volume = DicomVolume(
            study_instance_uid="CONSTANT-STUDY",
            series_instance_uid="CONSTANT-SERIES",
            volume= constant_array,
            source_path=Path(r"D:\test")
        )

    constant_result = normalizer.normalize(constant_volume)

    assert np.all(constant_result.volume == 0.0)

    print("Constant-intensity volume: PASS")
    
    print()
    print("=" * 70)
    print("INTENSITY NORMALIZER TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    main()