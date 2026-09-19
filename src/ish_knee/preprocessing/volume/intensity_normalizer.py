from ..dicom.dicom_volume import DicomVolume

import numpy as np

class IntensityNormalizer:

    """
    Normalizes MRI volume intensities using
    z-score normalization
    """

    def normalize(self, volume: DicomVolume)-> DicomVolume:
        """
        Normalize the voxel intensities of a DICOM volume.

        The resulting volume has approximately 
            mean = 0
            standard devitaion = 1

        Returns:
            A new DicomVolume containing the normalized data.
        """

        data = volume.volume.astype(
            np.float32,
            copy = False
        )

        mean = np.mean(data)

        std = np.std(data)

        # ----------------------------------------------------------
        # Handle constant volumes
        # ----------------------------------------------------------

        if std == 0:
            normalized = np.zeros_like(
                data,
                dtype=np.float32
            )

        else:
            normalized = (
                (data - mean) / std
            ).astype(
                np.float32
            )

        return DicomVolume(
            study_instance_uid=volume.series_instance_uid,
            series_instance_uid=volume.series_instance_uid,
            volume=normalized,
            source_path=volume.source_path
        )