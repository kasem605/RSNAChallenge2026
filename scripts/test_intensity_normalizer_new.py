
import numpy as np

from ish_knee.preprocessing.volume.intensity_normalizer import (
    IntensityNormalizer,
)


def main():
    normalizer = IntensityNormalizer()

    # Test 1: Normal volume
    volume = np.arange(
        20 * 32 * 32, dtype=np.float32
    ).reshape(20, 32, 32)

    original = volume.copy()
    normalized = normalizer.normalize(volume)

    assert normalized.shape == volume.shape
    assert normalized.dtype == np.float32
    assert normalized.flags.c_contiguous
    assert np.isfinite(normalized).all()
    assert np.array_equal(volume, original)
    assert abs(float(normalized.mean())) < 1e-5
    assert abs(float(normalized.std()) - 1.0) < 1e-4

    print("Normal volume test: PASS")

    # Test 2: Constant volume
    constant = np.full((8, 16, 16), 100.0, dtype=np.float32)
    result = normalizer.normalize(constant)

    assert np.all(result == 0)
    print("Constant volume test: PASS")

    # Test 3: Invalid dimensions
    try:
        normalizer.normalize(np.zeros((16, 16)))
        raise AssertionError("Expected ValueError for 2D input.")
    except ValueError:
        print("Invalid dimensions test: PASS")

    # Test 4: Non-finite values
    invalid = np.ones((8, 16, 16), dtype=np.float32)
    invalid[0, 0, 0] = np.nan

    try:
        normalizer.normalize(invalid)
        raise AssertionError("Expected ValueError for NaN.")
    except ValueError:
        print("Non-finite values test: PASS")

    print("\nALL INTENSITY NORMALIZER TESTS PASSED")


if __name__ == "__main__":
    main()
