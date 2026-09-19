from pathlib import Path

import numpy as np

from .dicom_series import DicomSeries
from .dicom_volume import DicomVolume

class DicomVolumeBuilder:

    """
    Builds a 3-D MRI volume from a DICOM series.
    """

    def build(self, series: DicomSeries) -> DicomVolume:

        """
        Converts the ordered DICOM slices into a 3-D numpy volume.
        """

        if not series.slices:
            raise ValueError("Cannot build volume from an empty DICOM series.")

        ordered_slices = series.ordered_slices()

        pixel_arrays = []

        for index, slice_ in enumerate(ordered_slices):

            if slice_.pixel_array is None:
                raise ValueError(f"Slice {index} contains no pixel data")

            pixel_arrays.append(slice_.pixel_array)

        volume = np.stack(
            pixel_arrays,
            axis=0
        )

        sorce_path = ordered_slices[0].path.parent

        return DicomVolume(
            study_instance_uid=series.study_instance_uid,
            series_instance_uid=series.series_instance_uid,
            volume=volume,
            source_path=sorce_path
        )