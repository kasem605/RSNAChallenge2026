
import numpy as np

from ish_knee.preprocessing.geometry.canonical_grid import (
    build_canonical_grid,
)
from ish_knee.preprocessing.geometry.volume_resampler import (
    VolumeResampler,
)


def test_identity_resampling():
    volume = np.arange(
        8 * 8 * 8,
        dtype=np.float32,
    ).reshape(8, 8, 8)

    source_affine = np.eye(4, dtype=np.float64)

    grid = build_canonical_grid(
        source_shape=volume.shape,
        source_affine=source_affine,
        spacing=1.0,
    )

    resampler = VolumeResampler()
    result = resampler.resample(
        volume,
        source_affine,
        grid,
    )

    assert result.shape == volume.shape
    assert np.allclose(result, volume, atol=1e-5)

    print("PASS: Identity resampling")


def test_spacing_change():
    # A constant image should remain constant wherever the
    # target grid overlaps the source volume.
    volume = np.full((8, 8, 8), 100.0, dtype=np.float32)

    source_affine = np.eye(4, dtype=np.float64)
    source_affine[:3, :3] *= 2.0

    grid = build_canonical_grid(
        source_shape=volume.shape,
        source_affine=source_affine,
        spacing=1.0,
    )

    resampler = VolumeResampler()
    result = resampler.resample(
        volume,
        source_affine,
        grid,
    )

    assert result.dtype == np.float32
    assert result.shape == grid.shape

    # Check the interior, away from the source boundaries.
    assert np.allclose(result[2:-2, 2:-2, 2:-2], 100.0)

    print("PASS: Spacing-change resampling")


if __name__ == "__main__":
    test_identity_resampling()
    test_spacing_change()
    print("\nVOLUME RESAMPLER TESTS PASSED")
