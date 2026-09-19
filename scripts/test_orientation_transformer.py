from pathlib import Path

import numpy as np

from ish_knee.preprocessing.dicom.dicom_volume import DicomVolume
from ish_knee.preprocessing.volume.orientation_transform import OrientationTransform
from ish_knee.preprocessing.volume.orientation_transformer import OrientationTransformer

def main():


    print("=" * 70)
    print("ORIENTATION TRANSFORMER TEST")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Create a known 3-D volume
    # ------------------------------------------------------------------------

    original = np.arange(
        24,
    ).reshape(
        2,
        3,
        4
    )

    volume = DicomVolume(
        study_instance_uid="TEST-STUDY",
        series_instance_uid="TEST-SERIES",
        volume=original,
        source_path=Path(r"D:\test")
    )

    transformer = OrientationTransformer()

    # ------------------------------------------------------------------------
    # Test 1: Axis permutation
    # ------------------------------------------------------------------------

    transformation = OrientationTransform(
        axis_order=(2, 1, 0),
        flip_axis_0=False,
        flip_axis_1=False,
        flip_axis_2=False,
    )

    result = transformer.transform(volume, transformation)

    expected = np.transpose(original, axes=(2, 1, 0))

    assert np.array_equal(result.volume, expected)

    print("Axis permutation test:       PASSED")

    # ------------------------------------------------------------------------
    # Test 2: Axis flip
    # ------------------------------------------------------------------------

    transformation = OrientationTransform(
            axis_order=(2, 1, 0),
            flip_axis_0=True,
            flip_axis_1=False,
            flip_axis_2=False,
        )

    result = transformer.transform(volume, transformation)

    expected = np.flip(expected, axis=0)
    
    assert np.array_equal(result.volume, expected)

    print("Axis flip test:     PASSED")

    # ------------------------------------------------------------------------
    # Test 3: Combined permutation + flips
    # ------------------------------------------------------------------------

    transformation = OrientationTransform(
        axis_order=(2, 1, 0),
        flip_axis_0=True,
        flip_axis_1=False,
        flip_axis_2=True,
    )

    result = transformer.transform(volume, transformation)

    # perform the same operations imdependently
    expected = original

    #first tyranspose\
    expected = np.transpose(original, axes=(2, 1, 0))

    # second: flip output axis 0
    expected = np.flip(expected, axis=0)

    # third: flip output axis 2
    expected = np.flip(expected, axis=2)

    print("Result shape:    ", result.volume.shape)
    print("Expected shape:    ", expected.shape)

    assert np.array_equal(result.volume, expected)

    print("Comdined transformation:     PASSED")

    # ------------------------------------------------------------------------
    # verify metadata
    # ------------------------------------------------------------------------    

    assert result.study_instance_uid == "TEST-STUDY"
    assert result.series_instance_uid == "TEST-SERIES"   
    assert result.source_path == Path(r"D:\test")

    print("Maintain preservation:   PASSED")

    # ------------------------------------------------------------------------
    # verify original laws was not modified
    # ------------------------------------------------------------------------  

    assert np.array_equal(volume.volume, original)

    print("Original volume unchanged:   PASSED")

    # ------------------------------------------------------------------------
    # verify contiguous memory
    # ------------------------------------------------------------------------  

    assert result.volume.flags["C_CONTIGUOUS"]

    print("Contiguous memory:       PASSED")

    print()
    print("ORIENTATION TRANSFORMER TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    main()

