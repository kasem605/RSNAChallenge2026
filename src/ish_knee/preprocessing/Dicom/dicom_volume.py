from dataclasses import dataclass
from pathlib import Path

import numpy as np

@dataclass
class DicomVolume:

    """
    Represents a 3-D MRI volume constructed from a DICOM series.
    """

    study_instance_uid: str
    series_instance_uid: str
    volume: np.ndarray
    source_path: Path

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Returns the dimensions of the 3-D volume.
        """
        
        return self.volume.shape


    @property
    def slice_count(self) -> int:

        """
        Returns the number of slices in the volume.
        """

        return self.volume.shape[0]