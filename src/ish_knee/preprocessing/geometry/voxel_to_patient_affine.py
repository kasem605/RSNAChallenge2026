
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VoxelToPatientAffine:
    """
    Maps NumPy voxel indices [slice, row, column]
    to DICOM patient coordinates [x, y, z] in LPS millimeters.

    patient_position = matrix @ voxel_index + origin
    """

    matrix: np.ndarray
    origin: np.ndarray

    def voxel_to_patient(
        self,
        voxel_index: tuple[float, float, float],
    ) -> np.ndarray:
        index = np.asarray(voxel_index, dtype=np.float64)

        if index.shape != (3,):
            raise ValueError(
                "voxel_index must contain (slice, row, column)."
            )

        return self.origin + self.matrix @ index

    def patient_to_voxel(
        self,
        patient_position: tuple[float, float, float],
    ) -> np.ndarray:
        position = np.asarray(patient_position, dtype=np.float64)

        if position.shape != (3,):
            raise ValueError(
                "patient_position must contain (x, y, z)."
            )

        return np.linalg.solve(
            self.matrix,
            position - self.origin,
        )


def calculate_voxel_to_patient_affine(series) -> VoxelToPatientAffine:
    """
    Calculate the affine from the ordered DICOM slices.

    Expects:
      series.ordered_slices()
      each slice to expose image_position and image_orientation

    Also expects each slice to expose pixel_spacing, or the series
    to expose pixel_spacing as a two-element (row, column) tuple.
    """

    slices = series.ordered_slices()

    if len(slices) < 2:
        raise ValueError(
            "At least two slices are required to estimate slice direction."
        )

    first = slices[0]

    if first.image_position is None:
        raise ValueError("Missing ImagePositionPatient.")

    if first.image_orientation is None:
        raise ValueError("Missing ImageOrientationPatient.")

    orientation = np.asarray(
        first.image_orientation,
        dtype=np.float64,
    )
    origin = np.asarray(first.image_position, dtype=np.float64)

    if orientation.shape != (6,):
        raise ValueError("ImageOrientationPatient must have 6 values.")

    if origin.shape != (3,):
        raise ValueError("ImagePositionPatient must have 3 values.")

    # DICOM: first triplet = increasing column index;
    # second triplet = increasing row index.
    column_direction = orientation[:3]
    row_direction = orientation[3:]

    column_norm = np.linalg.norm(column_direction)
    row_norm = np.linalg.norm(row_direction)

    if not np.isclose(column_norm, 1.0, atol=1e-4):
        raise ValueError("Invalid column direction cosine.")

    if not np.isclose(row_norm, 1.0, atol=1e-4):
        raise ValueError("Invalid row direction cosine.")

    if not np.isclose(
        np.dot(column_direction, row_direction),
        0.0,
        atol=1e-4,
    ):
        raise ValueError("DICOM in-plane directions are not orthogonal.")

    positions = np.asarray(
        [item.image_position for item in slices],
        dtype=np.float64,
    )

    if positions.shape != (len(slices), 3):
        raise ValueError("Invalid or missing slice positions.")

    position_steps = np.diff(positions, axis=0)

    # Estimate the actual 3D slice step vector. This preserves
    # small oblique components instead of assuming a purely normal step.
    slice_step = np.median(position_steps, axis=0)

    if np.linalg.norm(slice_step) <= 1e-6:
        raise ValueError("Slice positions do not define a valid step.")

    # PixelSpacing = (row spacing, column spacing).
    pixel_spacing = getattr(first, "pixel_spacing", None)

    if pixel_spacing is None:
        pixel_spacing = getattr(series, "pixel_spacing", None)

    if pixel_spacing is None or len(pixel_spacing) != 2:
        raise ValueError(
            "PixelSpacing (row, column) is required. "
            "Expose it on each DicomSlice or on DicomSeries."
        )

    row_spacing, column_spacing = map(float, pixel_spacing)

    if row_spacing <= 0 or column_spacing <= 0:
        raise ValueError("Pixel spacing must be positive.")

    # Matrix columns correspond to NumPy indices:
    # [slice, row, column].
    matrix = np.column_stack(
        (
            slice_step,
            row_direction * row_spacing,
            column_direction * column_spacing,
        )
    )

    if abs(np.linalg.det(matrix)) < 1e-8:
        raise ValueError("Voxel-to-patient affine is singular.")

    return VoxelToPatientAffine(
        matrix=matrix,
        origin=origin,
    )
