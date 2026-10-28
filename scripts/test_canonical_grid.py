
import numpy as np

from ish_knee.preprocessing.geometry.canonical_grid import (
    build_canonical_grid,
)


def main():
    # Synthetic volume with 2 mm spacing along each source axis.
    source_shape = (10, 20, 30)

    source_affine = np.eye(4, dtype=np.float64)
    source_affine[:3, :3] = np.diag([2.0, 2.0, 2.0])
    source_affine[:3, 3] = [-10.0, -20.0, 30.0]

    grid = build_canonical_grid(
        source_shape=source_shape,
        source_affine=source_affine,
        spacing=1.0,
    )

    print("CANONICAL GRID TEST")
    print("Source shape:", source_shape)
    print("Output shape:", grid.shape)
    print("Output spacing:", grid.spacing)
    print("Output affine:\n", grid.affine)

    assert grid.shape == (19, 39, 59)
    assert np.allclose(grid.spacing, (1.0, 1.0, 1.0))
    assert np.allclose(
        grid.affine[:3, :3],
        np.eye(3),
    )

    print("\nCANONICAL GRID TEST PASSED")


if __name__ == "__main__":
    main()
