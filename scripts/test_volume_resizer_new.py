
import numpy as np

from ish_knee.preprocessing.volume.volume_resizer import (
    VolumeResizer,
    VolumeResizeConfig,
)


def main():
    print("=" * 70)
    print("VOLUME RESIZER TEST")
    print("=" * 70)

    resizer = VolumeResizer(
        VolumeResizeConfig(
            target_shape=(96, 160, 160)
        )
    )

    # Test 1: Resize a representative 3D volume.
    volume = np.random.default_rng(42).random(
        (22, 128, 128),
        dtype=np.float32,
    )

    original = volume.copy()
    resized = resizer.resize(volume)

    assert resized.shape == (96, 160, 160)
    print("Target shape: PASS")

    assert resized.dtype == np.float32
    print("Float32 output: PASS")

    assert resized.flags.c_contiguous
    print("Contiguous memory: PASS")

    assert np.isfinite(resized).all()
    print("Finite values: PASS")

    assert np.array_equal(volume, original)
    print("Original volume unchanged: PASS")

    assert resized.min() >= -1e-6
    assert resized.max() <= 1.0 + 1e-6
    print("Intensity range: PASS")

    # Test 2: Constant-intensity volume.
    constant = np.full(
        (8, 16, 16),
        0.5,
        dtype=np.float32,
    )

    constant_resized = resizer.resize(constant)

    assert constant_resized.shape == (96, 160, 160)
    assert np.allclose(constant_resized, 0.5, atol=1e-6)
    print("Constant volume: PASS")

    # Test 3: Reject invalid dimensionality.
    try:
        resizer.resize(np.zeros((16, 16), dtype=np.float32))
        raise AssertionError("Expected ValueError for 2D input.")
    except ValueError:
        print("Invalid dimensions: PASS")

    # Test 4: Reject non-finite input.
    invalid = np.ones((8, 16, 16), dtype=np.float32)
    invalid[0, 0, 0] = np.nan

    try:
        resizer.resize(invalid)
        raise AssertionError("Expected ValueError for NaN.")
    except ValueError:
        print("Non-finite input: PASS")

    print("\nALL VOLUME RESIZER TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
