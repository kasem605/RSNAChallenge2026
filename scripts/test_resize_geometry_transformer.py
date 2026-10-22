
import numpy as np

from ish_knee.preprocessing.geometry.resize_geometry_transformer import (
    ResizeGeometryTransformer,
)


def main():
    print("=" * 70)
    print("RESIZE GEOMETRY TRANSFORMER TEST")
    print("=" * 70)

    source_shape = (114, 177, 179)
    target_shape = (96, 160, 160)

    source_affine = np.eye(4, dtype=np.float64)
    source_affine[:3, :3] = np.diag([1.0, 1.2, 0.8])
    source_affine[:3, 3] = [10.0, 20.0, 30.0]

    geometry = ResizeGeometryTransformer.transform(
        source_shape=source_shape,
        source_affine=source_affine,
        target_shape=target_shape,
    )

    assert geometry.shape == target_shape
    assert geometry.affine.shape == (4, 4)
    assert np.isfinite(geometry.affine).all()
    print("Output geometry validation: PASS")

    # Check several target voxel centers against the mapping
    # used by grid_mode=True resizing.
    scale = np.asarray(source_shape) / np.asarray(target_shape)

    test_indices = [
        np.array([0.0, 0.0, 0.0]),
        (np.asarray(target_shape) - 1) / 2.0,
        np.asarray(target_shape, dtype=float) - 1.0,
    ]

    for target_index in test_indices:
        source_index = (
            scale * (target_index + 0.5) - 0.5
        )

        expected_patient = (
            source_affine[:3, :3] @ source_index
            + source_affine[:3, 3]
        )

        homogeneous_target = np.append(target_index, 1.0)
        actual_patient = (
            geometry.affine @ homogeneous_target
        )[:3]

        assert np.allclose(
            actual_patient,
            expected_patient,
            atol=1e-8,
        )

    print("Voxel-to-patient coordinate mapping: PASS")

    # Confirm that the affine remains invertible.
    assert abs(np.linalg.det(geometry.affine[:3, :3])) > 1e-12
    print("Affine invertibility: PASS")

    print("\nALL RESIZE GEOMETRY TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
