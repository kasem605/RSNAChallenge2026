import numpy as np

from dataclasses import dataclass

from ..dicom.dicom_volume import DicomVolume
from ..dicom.dicom_series import DicomSeries
from .orientation_info import OrientationInfo

@dataclass(frozen=True)
class OrientationAnalyzer:
    """
    Analyzes the geometric orientation of a DICOM series

    This class does not modify the image volume.
    """

    def analyze(self, series: DicomSeries) -> OrientationInfo:

        if not series.slices:
            raise ValueError(
                "cannot analyze orientation of an empty DICOM series."
            )

        first_slice = series.slices[0]

        if first_slice.image_position is None:
            raise ValueError(
                "DICOM series does not contain ImageOrientationPatient."
            )

        orientation = first_slice.image_orientation

        if len(orientation) != 6:
            raise ValueError(
                "image orientation must contain 6 values"
            )

        row_direction = np.array(
            orientation[0:3],
            dtype=float
        )

        column_direction = np.array(
            orientation[3:6],
            dtype=float
        )

        slice_normal = np.cross(
            row_direction,
            column_direction
        )

        anatomical_plane = self._determine_plane(
            slice_normal
        )

        return OrientationInfo(
            row_direction=(
                float(row_direction[0]),
                float(row_direction[1]),
                float(row_direction[2]),
            ),
            column_direction=(
                float(row_direction[0]),
                float(row_direction[1]),
                float(row_direction[2]),
            ),
            slice_normal=(
                float(row_direction[0]),
                float(row_direction[1]),
                float(row_direction[2]),
            ),
            anatomical_plane=anatomical_plane
        )

    @staticmethod
    def _determine_plane(slice_normal: np.ndarray) -> str:

        """
        Determine the primary anatomical plane from
        the slice normal.

        DICOM patient coordinates are treated as:
            X = left/right
            Y = anterior/posterior
            Z = inferior/superior

        therefore:
        X-dominant normal -> sagittal
        Y-dominant normal -> coronal
        Z-dominant normal -> axial

        """

        absolute_normal = np.abs(slice_normal)

        dominant_axis = int(
            np.argmax(absolute_normal)
        )

        if dominant_axis == 0:
            return "sagittal"

        if dominant_axis == 1:
            return "coronal"

        return "axial"