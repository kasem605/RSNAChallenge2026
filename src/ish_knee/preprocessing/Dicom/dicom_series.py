from dataclasses import dataclass

from .dicom_slice import DicomSlice

import numpy as np


@dataclass(frozen=True)
class DicomSeries:
    """
    Represents a collection of DICOM slices
    belonging to one MRI series.
    """

    study_instance_uid: str
    series_instance_uid: str
    slices: tuple[DicomSlice, ...]

    @property
    def image_count(self) -> int:
        """
        Number of DICOM images in the series.
        """
        return len(self.slices)

    @property
    def image_shape(self) -> tuple[int, int] | None:
        """
        Shape of the image data.

        Returns the shape of the first slice.
        Assumes all slices in the series have
        the same dimensions.

        Returns:
            (rows, columns), or None if no pixel data exists.
        """
        if not self.slices:
            return None

        first_slice = self.slices[0]

        if first_slice.pixel_array is None:
            return None

        return first_slice.pixel_array.shape

    def ordered_slices(self) -> tuple[DicomSlice, ...]:
        """
        Return the DICOM slices in spatial order.

        When ImagePositionPatient and ImagePositionPatient
        are available, ordering is based on the slice normal.

        InstanceNumber is used as a fallback
        """

        if not self.slices:
            return ()

        # -----------------------------------------------------------------------------------
        # Check whether we have the geometry required
        # for spatial ordering
        # -----------------------------------------------------------------------------------

        has_geometry = all(
            slice_.image_position is not None
            and slice_.image_orientation is not None
            for slice_ in self.slices
        )

        if not has_geometry:
            return tuple(
                sorted(
                    self.slices,
                    key=lambda slice_: slice_.instance_number
                )
            )
        
        # -----------------------------------------------------------------------------------
        # Get orientation from the first slice.
        # -----------------------------------------------------------------------------------

        orientation = self.slices[0].image_orientation

        if orientation is None:
            return tuple(
                sorted(
                    self.slices,
                    key=lambda slice_: slice_.instance_number
                )
            )

        row_direction = np.array(
            orientation[0:3],
            dtype=float
        )

        column_direction = np.array(
            orientation[3:6],
            dtype=float
        )

        # -----------------------------------------------------------------------------------
        # calculate slice normal
        #
        # Normal = row direction x column direction
        # -----------------------------------------------------------------------------------

        slice_normal = np.cross(
            row_direction,
            column_direction
        )

        # -----------------------------------------------------------------------------------
        # calculate spatial position of each slice
        #
        # Projection of ImagePositionPatient onto
        # the slice normal gives the slice location.
        # -----------------------------------------------------------------------------------       

        def slice_location( slice_: DicomSlice) -> float:

            position = np.array(
                slice_.image_position,
                dtype=float
            )

            return float(np.dot(position, slice_normal))

        return tuple(
            sorted(
                self.slices,
                key=slice_location
            )
        )