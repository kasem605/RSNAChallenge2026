from dataclasses import dataclass

from .dicomslice import DicomSlice

@dataclass(frozen=True)
class DicomSeries:
    """
    Represents a DICOM series
    """

    study_instance_uid: str
    series_instance_uid: str
    slices: tuple[DicomSlice, ...]

    @property
    def slice_count(self) -> int:
        return len(self.slices)

    @property
    def shape(self) -> tuple[int, int]:
        if not self.slices:
            return (0,0)

        return self.slices[0].pixel_array.shape

    def ordered_slices(self) -> tuple[DicomSlice, ...]:
        """
        Return slices ordered by ImagePositionPatient
        when available, otherwise by InstanceNumber
        """

        if not self.slices:
            return()

        if all(
            slice_.image_position is not None
            for slice_ in self.slices ):
            return tuple(
                sorted(
                    self.slices,
                    key=lambda slice_: slice_.image_position
                )
            )

        return tuple(
            sorted(
                self.slices,
                key=lambda slice_: slice_.instance_number
            )
        )