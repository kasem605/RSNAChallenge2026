
from ..dicom.dicom_volume import DicomVolume

import numpy as np


class IntensityNormalizer:
    """
    Normalize voxel intensities in a DICOM volume to [0, 1].

    The input DicomVolume is not modified.

    Axis convention:
        Axis 0 = depth / slice
        Axis 1 = row / height
        Axis 2 = column / width

    Behavior:
        - Returns a new DicomVolume.
        - Converts voxel data to float32.
        - Handles constant-intensity volumes.
        - Rejects NaN and infinite values.
        - Preserves study UID, series UID, and source path.
    """

    def normalize(self, volume: DicomVolume) -> DicomVolume:
        """
        Normalize a DicomVolume using min-max normalization.

        Formula:
            normalized = (voxel - minimum) / (maximum - minimum)

        Returns:
            A new DicomVolume containing normalized float32 voxels.

        Raises:
            TypeError: If volume is not a DicomVolume.
            ValueError: If the volume is not 3D, is empty,
                        or contains non-finite values.
        """

        # Validate input type
        if not isinstance(volume, DicomVolume):
            raise TypeError(
                "Expected a DicomVolume instance."
            )

        data = volume.volume

        # Validate dimensionality
        if data.ndim != 3:
            raise ValueError(
                f"MRI volume must be 3D; received {data.ndim}D."
            )

        # Validate non-empty input
        if data.size == 0:
            raise ValueError(
                "MRI volume cannot be empty."
            )

        # Validate numeric data
        if not np.issubdtype(data.dtype, np.number):
            raise TypeError(
                "MRI volume must contain numeric values."
            )

        # Convert to float32 without modifying the original volume
        data = data.astype(np.float32, copy=True)

        # Reject NaN and infinite values
        if not np.isfinite(data).all():
            raise ValueError(
                "MRI volume contains NaN or infinite values."
            )

        # Find intensity range
        minimum = float(np.min(data))
        maximum = float(np.max(data))

        # Handle constant-intensity volumes
        if maximum == minimum:
            normalized = np.zeros_like(
                data,
                dtype=np.float32,
            )
        else:
            # Min-max normalization to [0, 1]
            normalized = (data - minimum) / (
                maximum - minimum
            )

        # Ensure contiguous float32 output
        normalized = np.ascontiguousarray(
            normalized,
            dtype=np.float32,
        )

        # Return a new volume while preserving metadata
        return DicomVolume(
            study_instance_uid=volume.study_instance_uid,
            series_instance_uid=volume.series_instance_uid,
            volume=normalized,
            source_path=volume.source_path,
        )
