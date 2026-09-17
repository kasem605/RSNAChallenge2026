from dataclasses import dataclass
import numpy as np

from ..dicom.dicom_volume import DicomVolume
from .orientation_info import OrientationInfo

@dataclass
class OrientationTransformer:

    """
    Transforms a DICOM volume into a consistent orientation.

    The initial implementation prserves the existing volume
    orientation.  Actual axis transformations will be added after
    the target orientation convention has been verified.
    """

    def transform(
            self,
            volume: DicomVolume,
            orientation: OrientationInfo
    ) -> DicomVolume:

        if not isinstance(volume, DicomVolume):
            raise TypeError("Expected a DicomVolume instance.")

        if not isinstance(orientation, OrientationInfo):
            raise TypeError("Expected an OrientationInfo instance.")

        if volume.volume.ndim != 3:
            raise ValueError("MRI volume must be 3-dimensional.")

        transformed_volume = np.array(
            volume.volume,
            copy=True
        )

        return DicomVolume(
            study_instance_uid=volume.study_instance_uid,
            series_instance_uid=volume.series_instance_uid,
            volume=transformed_volume,
            source_path=volume.source_path
        )
