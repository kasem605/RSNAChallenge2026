
from ..dicom.dicom_volume import DicomVolume
from .preprocessing_config import PreprocessingConfig
from .spacing_resampler import SpacingResampler
from .volume_padder import VolumePadder
from .intensity_normalizer import IntensityNormalizer

from .preprocessing_diagnostics import ShapeDiagnostics

import numpy as np


class VolumePreprocessor:
    """
    Preprocess a 3-D MRI volume while preserving voxel spacing.

    Processing order:
        1. Validate input
        2. Resample to target voxel spacing
        3. Center-crop dimensions that exceed target shape
        4. Pad smaller dimensions
        5. Normalize intensity
    """

    def __init__(self, config: PreprocessingConfig) -> None:
        if not isinstance(config, PreprocessingConfig):
            raise TypeError(
                "config must be a PreprocessingConfig instance"
            )

        self._config = config
        self._spacing_resampler = SpacingResampler()
        self._volume_padder = VolumePadder()
        self._intensity_normalizer = IntensityNormalizer()

    @staticmethod
    def _center_crop(
        volume: DicomVolume,
        target_shape: tuple[int, int, int],
    ) -> DicomVolume:
        """Center-crop each dimension without resizing voxel values."""

        array = volume.volume

        if array.ndim != 3:
            raise ValueError("MRI volume must be three-dimensional")

        if len(target_shape) != 3 or any(
            dimension <= 0 for dimension in target_shape
        ):
            raise ValueError("target_shape must contain three positive dimensions")

        slices = []

        for current, target in zip(array.shape, target_shape):
            if current <= target:
                slices.append(slice(0, current))
            else:
                start = (current - target) // 2
                slices.append(slice(start, start + target))

        cropped = np.ascontiguousarray(array[tuple(slices)])

        return DicomVolume(
            study_instance_uid=volume.study_instance_uid,
            series_instance_uid=volume.series_instance_uid,
            volume=cropped,
            source_path=volume.source_path,
        )

    
    def preprocess(
        self,
        volume: DicomVolume,
        current_spacing,
    ) -> DicomVolume:

        if not isinstance(volume, DicomVolume):
            raise TypeError("volume must be a DicomVolume instance")

        original_shape = tuple(volume.volume.shape)

        # Step 1: Resample to target physical voxel spacing.
        processed_volume = self._spacing_resampler.resample(
            volume=volume,
            current_spacing=current_spacing,
            target_spacing=self._config.target_spacing,
        )

        resampled_shape = tuple(processed_volume.volume.shape)
        target_shape = self._config.target_shape

        if len(resampled_shape) != 3:
            raise ValueError("Resampled MRI volume must be three-dimensional")

        # Report dimensions before cropping or padding.
        diagnostics = ShapeDiagnostics(
            original_shape=original_shape,
            resampled_shape=resampled_shape,
            target_shape=target_shape,
        )
        diagnostics.print_report()

        # Step 2: Center-crop oversized dimensions.
        if any(
            current > target
            for current, target in zip(resampled_shape, target_shape)
        ):
            processed_volume = self._center_crop(
                processed_volume,
                target_shape,
            )

        # Step 3: Symmetrically pad undersized dimensions.
        if processed_volume.volume.shape != target_shape:
            processed_volume = self._volume_padder.pad(
                volume=processed_volume,
                target_shape=target_shape,
                constant_value=self._config.padding_value,
            )

        # Step 4: Normalize intensity.
        if self._config.normalize_intensity:
            processed_volume = self._intensity_normalizer.normalize(
                volume=processed_volume
            )

        # Final validation.
        if processed_volume.volume.shape != target_shape:
            raise ValueError(
                f"Preprocessed shape is {processed_volume.volume.shape}; "
                f"expected {target_shape}"
            )

        if not np.isfinite(processed_volume.volume).all():
            raise ValueError(
                "Preprocessed volume contains NaN or infinite values"
            )

        return processed_volume
