from dataclasses import dataclass
from pathlib import Path

from .dicomslice import DicomSlice

@dataclass(frozen=True)
class DicomSeries:
    """
    Represents a DICOM series
    """

    series_path: Path
    slices: tuple[DicomSlice, ...]

    @property
    def slice_count(self) -> int:
        return len(self.slices)

    @property
    def shape(self) -> tuple[int, int]:
        if not self.slices:
            return (0,0)

        return self.slices[0].pixel_array.shape