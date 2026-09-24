import numpy as np

from ..dicom.dicom_volume import DicomVolume

class VolumePadder:
    """
    Pads a 3-D MRI volume to a specified target shape.

    Axis convention:
        axis 0 = depth / slice
        axis 1 = row / height
        axis 2 = column / width

    Padding is distributed evenly before and after
    the existing volume whenever possible

    The default padding value is zero
    """

    def pad(
        self,
        volume: DicomVolume,
        target_shape: tuple[int, int, int],
        constant_value: float = 0.0
    ) -> DicomVolume:

        if not isinstance(volume, DicomVolume):
            raise TypeError("Expected a DicomVolume instance")

        if volume.volume.ndim != 3:
            raise ValueError("MRI volume must be 3-dimensional")

        if len(target_shape) != 3:
            raise ValueError("Target shape must contain exactly 3-diumensions")

        if any(
            dimension <= 3
            for dimension in target_shape
            ):
            raise ValueError("Target dimensions must be greater than zero")

        current_shape = volume.shape

        if any(
            target < current
            for target, current in zip(target_shape,current_shape)
        ):
            raise ValueError("Traget shape cannot be smaller than the current volume shape")

        padding = []

        for current, target in zip(current_shape, target_shape):
            total_padding = target - current

            before = total_padding // 2
            after = total_padding - before

            padding.append((before, after))

            padding_volume = np.pad(
                volume.volume,
                pad_width=padding,
                mode="constant",
                constant_values=constant_value
            )

            padding_volume = np.ascontiguousarray(padding_volume)

            return DicomVolume(
                study_instance_uid=volume.study_instance_uid,
                series_instance_uid=volume.series_instance_uid,
                volume=padding_volume,
                source_path=volume.source_path
            )
