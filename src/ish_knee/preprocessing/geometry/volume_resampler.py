
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import affine_transform

from .canonical_grid import CanonicalGrid


@dataclass
class VolumeResampler:
    """
    Resamples a 3D MRI volume onto a canonical output grid.

    The source affine maps source NumPy indices [slice, row, column]
    to patient LPS coordinates.

    The output grid affine maps output indices to patient LPS coordinates.
    """

    interpolation_order: int = 1
    fill_value: float = 0.0

    def __post_init__(self):
        if self.interpolation_order not in (0, 1, 3):
            raise ValueError(
                "interpolation_order must be 0, 1, or 3."
            )

    def resample(
        self,
        volume: np.ndarray,
        source_affine: np.ndarray,
        target_grid: CanonicalGrid,
    ) -> np.ndarray:
        """
        Return a resampled float32 volume.

        Parameters
        ----------
        volume:
            Three-dimensional source MRI array.
        source_affine:
            4x4 source voxel-to-patient LPS affine.
        target_grid:
            CanonicalGrid describing the output shape and affine.
        """

        volume = np.asarray(volume)
        source_affine = np.asarray(
            source_affine,
            dtype=np.float64,
        )

        if volume.ndim != 3:
            raise ValueError("Source volume must be 3D.")

        if any(size <= 0 for size in volume.shape):
            raise ValueError("Source volume dimensions must be positive.")

        if not np.issubdtype(volume.dtype, np.number):
            raise ValueError("Source volume must contain numeric data.")

        if not np.all(np.isfinite(volume)):
            raise ValueError("Source volume contains non-finite values.")

        if source_affine.shape != (4, 4):
            raise ValueError("source_affine must have shape (4, 4).")

        if not np.all(np.isfinite(source_affine)):
            raise ValueError("source_affine contains non-finite values.")

        if not np.allclose(
            source_affine[3],
            [0, 0, 0, 1],
            atol=1e-8,
        ):
            raise ValueError("source_affine must be homogeneous.")

        if abs(np.linalg.det(source_affine[:3, :3])) < 1e-10:
            raise ValueError("source_affine is singular.")

        # Map target voxel indices to source voxel indices:
        #
        # source_index =
        # inverse(source_affine) @ target_affine @ target_index
        #
        # scipy expects this output-to-input mapping.
        output_to_source = (
            np.linalg.inv(source_affine)
            @ target_grid.affine
        )

        matrix = output_to_source[:3, :3]
        offset = output_to_source[:3, 3]

        result = affine_transform(
            input=volume.astype(np.float32, copy=False),
            matrix=matrix,
            offset=offset,
            output_shape=target_grid.shape,
            order=self.interpolation_order,
            mode="constant",
            cval=float(self.fill_value),
            prefilter=self.interpolation_order > 1,
        )

        return np.ascontiguousarray(result, dtype=np.float32)
