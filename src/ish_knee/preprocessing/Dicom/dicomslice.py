from dataclasses import dataclass
from pathlib import Path

import numpy as np

@dataclass(frozen=True)
class DicomSlice:
    """
    Represents a single DICOM slice
    """

    path: Path
    instance_number: int
    image_position: tuple[float,float,float] | None
    pixel_array: np.ndarray | None