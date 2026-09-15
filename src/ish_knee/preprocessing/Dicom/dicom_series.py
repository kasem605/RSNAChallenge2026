from dataclasses import dataclass

from .dicomslice import DicomSlice


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

        ImagePositionPatient is preferred when available.
        InstanceNumber is used as a fallback.
        """

        if not self.slices:
            return ()

        # Preferred ordering when every slice has
        # ImagePositionPatient.
        if all(
            slice_.image_position is not None
            for slice_ in self.slices
        ):
            return tuple(
                sorted(
                    self.slices,
                    key=lambda slice_: slice_.image_position
                )
            )

        # Fallback ordering.
        return tuple(
            sorted(
                self.slices,
                key=lambda slice_: slice_.instance_number
            )
        )

