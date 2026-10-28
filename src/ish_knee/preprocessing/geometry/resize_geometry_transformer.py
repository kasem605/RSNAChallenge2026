
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResizedVolumeGeometry:
    """Geometry metadata for a resized 3D volume."""

    shape: tuple[int, int, int]
    affine: np.ndarray


class ResizeGeometryTransformer:
    """
    Calculate voxel-to-patient geometry after volume resizing.

    Assumptions:
        - The source affine maps voxel-center indices to patient space.
        - Resizing uses grid_mode=True.
        - Source and target grids cover the same voxel-boundary extent.
        - Axis order is unchanged by resizing.
    """

    @staticmethod
    def transform(
        source_shape: tuple[int, int, int],
        source_affine: np.ndarray,
        target_shape: tuple[int, int, int],
    ) -> ResizedVolumeGeometry:

        if len(source_shape) != 3 or len(target_shape) != 3:
            raise ValueError("Source and target shapes must be 3D.")

        if any(size <= 0 for size in source_shape + target_shape):
            raise ValueError("All dimensions must be positive.")

        source_affine = np.asarray(
            source_affine,
            dtype=np.float64,
        )

        if source_affine.shape != (4, 4):
            raise ValueError("Source affine must be a 4x4 matrix.")

        if not np.isfinite(source_affine).all():
            raise ValueError("Source affine contains non-finite values.")

        if not np.allclose(
            source_affine[3],
            [0.0, 0.0, 0.0, 1.0],
        ):
            raise ValueError("Invalid homogeneous affine last row.")

        if abs(np.linalg.det(source_affine[:3, :3])) < 1e-12:
            raise ValueError("Source affine is not invertible.")

        scale = np.asarray(source_shape, dtype=np.float64) / np.asarray(
            target_shape,
            dtype=np.float64,
        )

        # Under grid_mode=True, target voxel centers map to source
        # coordinates as:
        # source_index = scale * (target_index + 0.5) - 0.5
        #
        # Therefore:
        # source_index = scale * target_index + (scale - 1) / 2
        index_transform = np.eye(4, dtype=np.float64)
        index_transform[:3, :3] = np.diag(scale)
        index_transform[:3, 3] = (scale - 1.0) / 2.0

        # Map target indices through source index space into patient space.
        target_affine = source_affine @ index_transform

        return ResizedVolumeGeometry(
            shape=tuple(target_shape),
            affine=target_affine,
        )
