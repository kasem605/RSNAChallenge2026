
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DicomSlice:
    """
    Represents one DICOM image slice from an MRI series.

    Attributes:
        path: Source DICOM file path.
        instance_number: DICOM InstanceNumber, used as a fallback
            when spatial ordering metadata is unavailable.
        image_position: ImagePositionPatient (x, y, z), in mm.
        image_orientation: ImageOrientationPatient containing
            the first and second direction-cosine vectors.
        pixel_array: Decoded 2D image data, or None if unavailable.
    """

    path: Path
    instance_number: int
    image_position: tuple[float, float, float] | None
    image_orientation: tuple[
        float, float, float,
        float, float, float
    ] | None
    pixel_array: np.ndarray | None
    pixel_spacing: tuple[float, float] | None
