import numpy as np

from ..dicom.dicom_volume import DicomVolume

class VolumeCropper:

    """
    Crops a 3-D MRI volume using explicit axis ranges

    Axis convention:
        axis 0 = depth / slice
        axis 1 = row / height
        axis 2 = column /width

    Crop ranges use Python slice semantics:
        (start, stop)

    Example:
        Crop(volume, (1,4), (10, 100), (20, 120))
    """

    def crop(
            self,
            volume: DicomVolume,
            axis_0_range: tuple[int, int],
            axis_1_range: tuple[int, int],
            axis_2_range: tuple[int, int]
    ) -> DicomVolume:

        if not isinstance(volume,DicomVolume):
            raise TypeError("Expected a DicomVolume instance")

        if volume.volume.ndim !=3:
            raise ValueError("MRI volume must be 3-dimensional.")

        ranges = (
            axis_0_range,
            axis_1_range,
            axis_2_range
        )

        for axis_range in ranges:
            if len(axis_range) != 2:
                raise ValueError("Each crop range must contain exactly two values.")

            start, stop = axis_range

            if start < 0 or stop < 0:
                raise ValueError("Crop range values cannot be negative.")

            if start >= stop:
                raise ValueError("Crop range start must be less than stop.")

        volume_shape = volume.volume.shape

        for axis, axis_range in enumerate(ranges):
            _, stop = axis_range

            if stop > volume_shape[axis]:
                raise ValueError(f"Crop range exceeds volume dimension on axis {axis}")

        cropped_volume = volume.volume[
            axis_0_range[0]:axis_0_range[1],
            axis_1_range[0]:axis_1_range[1],
            axis_2_range[0]:axis_2_range[1]
        ]

        cropped_volume = np.ascontiguousarray(cropped_volume)

        return DicomVolume(
            study_instance_uid=volume.study_instance_uid,
            series_instance_uid=volume.series_instance_uid,
            volume=cropped_volume,
            source_path=volume.source_path
        )
