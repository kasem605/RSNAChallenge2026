from ..dicom.dicom_volume import DicomVolume

import numpy as np

class IntensityNormalizer:

    """
    Normalize the voxel intensities of a DICOM volume.

    Default behavior:
        Normalize the volume to the range [0,1]

    Axis convention:
        Axis 0 = depth / slice
        Axis 1 = row / height
        Axis 2 = column / width
    """

    def normalize(self, volume: DicomVolume) -> DicomVolume:

        if not isinstance(volume, DicomVolume):
            raise TypeError("Expected a DicomVolume instance")

        if volume.volume.ndim !=3:
            raise ValueError("MRI volume must be 3-dimensionals")

        data = volume.volume.astype(
            np.float32,
            copy=False
        )

        minimum = np.min(data)
        maximum = np.max(data)

        # ----------------------------------------------------------
        # Handle a constant-intensity volume
        # ----------------------------------------------------------

        if maximum == minimum:

            normalized = np.zeros_like(data, dtype=np.float32)
        else:

            normalized = ((data - minimum) / (maximum - minimum))

        normalized= np.ascontiguousarray(normalized, dtype=np.float32)

        return DicomVolume(
            study_instance_uid=volume.study_instance_uid,
            series_instance_uid=volume.series_instance_uid,
            volume=normalized,
            source_path=volume.source_path
        )