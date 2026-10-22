
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CanonicalGrid:
    """
    Describes a regular 3D output grid in DICOM LPS coordinates.

    Array axes are ordered as [axis_0, axis_1, axis_2].
    The affine maps those voxel indices to patient coordinates.
    """

    shape: tuple[int, int, int]
    affine: np.ndarray
    spacing: tuple[float, float, float]

    def __post_init__(self):
        if len(self.shape) != 3 or any(n <= 0 for n in self.shape):
            raise ValueError("shape must contain three positive dimensions.")

        if len(self.spacing) != 3 or any(s <= 0 for s in self.spacing):
            raise ValueError("spacing must contain three positive values.")

        affine = np.asarray(self.affine, dtype=np.float64)

        if affine.shape != (4, 4):
            raise ValueError("affine must have shape (4, 4).")

        if not np.all(np.isfinite(affine)):
            raise ValueError("affine contains non-finite values.")

        if not np.allclose(affine[3], [0, 0, 0, 1]):
            raise ValueError("affine must be a homogeneous 4x4 matrix.")

        if abs(np.linalg.det(affine[:3, :3])) < 1e-10:
            raise ValueError("affine is singular.")

        object.__setattr__(self, "affine", affine)


def build_canonical_grid(
    source_shape: tuple[int, int, int],
    source_affine: np.ndarray,
    spacing: float = 1.0,
) -> CanonicalGrid:
    """
    Build an axis-aligned LPS grid covering the physical extent of
    a source volume.

    Parameters
    ----------
    source_shape:
        Source NumPy shape, ordered [slice, row, column].
    source_affine:
        4x4 voxel-to-patient LPS affine for that same axis order.
    spacing:
        Isotropic output voxel spacing in millimeters.

    Notes
    -----
    The grid covers the source voxel-center extent, not the outer
    half-voxel boundaries. Oblique source volumes are bounded using
    the eight corners of their voxel-center extent.
    """

    if len(source_shape) != 3 or any(n <= 0 for n in source_shape):
        raise ValueError("source_shape must contain three positive dimensions.")

    if spacing <= 0 or not np.isfinite(spacing):
        raise ValueError("spacing must be a finite positive number.")

    source_affine = np.asarray(source_affine, dtype=np.float64)

    if source_affine.shape != (4, 4):
        raise ValueError("source_affine must have shape (4, 4).")

    if not np.all(np.isfinite(source_affine)):
        raise ValueError("source_affine contains non-finite values.")

    if not np.allclose(source_affine[3], [0, 0, 0, 1]):
        raise ValueError("source_affine must be homogeneous.")

    if abs(np.linalg.det(source_affine[:3, :3])) < 1e-10:
        raise ValueError("source_affine is singular.")

    # Eight corners in source voxel coordinates.
    corners = np.array([
        [i, j, k, 1.0]
        for i in (0, source_shape[0] - 1)
        for j in (0, source_shape[1] - 1)
        for k in (0, source_shape[2] - 1)
    ], dtype=np.float64)

    patient_corners = (source_affine @ corners.T).T[:, :3]

    # DICOM LPS directions:
    # +X = Left, +Y = Posterior, +Z = Superior.
    minimum = patient_corners.min(axis=0)
    maximum = patient_corners.max(axis=0)

    # Use positive patient-coordinate directions for all output axes.
    output_origin = minimum
    output_shape = tuple(
        int(np.ceil((maximum[axis] - minimum[axis]) / spacing)) + 1
        for axis in range(3)
    )

    output_affine = np.eye(4, dtype=np.float64)
    output_affine[:3, :3] = np.diag([spacing, spacing, spacing])
    output_affine[:3, 3] = output_origin

    return CanonicalGrid(
        shape=output_shape,
        affine=output_affine,
        spacing=(spacing, spacing, spacing),
    )
