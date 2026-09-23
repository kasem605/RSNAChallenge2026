import numpy as np
from scipy.ndimage import zoom

from ..dicom.dicom_volume import DicomVolume

class VolumeResizer:
    """
    Rsizes 3-D MRI volume to a specified target shape.

    Target shape:
    (depth, height, width)
    """

    def resize(self, volume: DicomVolume, target_shape: tuple[int, int, int]) -> DicomVolume:
        if not isinstance( volume, DicomVolume):
            raise TypeError("Expected a DicomVolume instance")

        if volume.volume.ndim != 3:
            raise ValueError("MRI volume must be 3-dimenaional")

        if len(target_shape) != 3:
            raise ValueError("Target sshape must contain exactly three dimensions")

        if any(
            dimension <= 0
            for dimension in target_shape
        ):
            raise ValueError("Target dimensions must be greater than zero")

        current_shape = volume.volume.shape

        zoom_factors = tuple(
            target / current
            for target, current
            in zip(target_shape, current_shape)
            )

        resized_volume = zoom(
            volume.volume,
            zoom = zoom_factors,
            order=1
        )

        resized_volume = np.ascontiguousarray(resized_volume)

        return DicomVolume(
            study_instance_uid=volume.study_instance_uid,
            series_instance_uid=volume.series_instance_uid,
            volume = resized_volume,
            source_path=volume.source_path
        )
