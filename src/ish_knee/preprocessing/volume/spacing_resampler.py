import numpy as np
from scipy.ndimage import zoom

from ..dicom.dicom_volume import DicomVolume
from .voxel_spacing import VoxelSpacing

class ScacingResampler:

    """
    Resamples a 3-D MRI volume from its current physical
    voxel spacing to a target physical voxel spacing

    Axis xonvention:
        axis 0 = depth / slice
        axis 1 = row / height
        axis 2 = column / width

    Spacing values are expressed in millimeters

    """

    def resample(slef, volume: DicomVolume, current_spacing: VoxelSpacing, target_spacing: VoxelSpacing) -> DicomVolume:

        if not isinstance(volume, DicomVolume):
            raise TypeError("Expected DICOM volume instance")

        if not isinstance(current_spacing, VoxelSpacing):
            raise TypeError("Expected current_spacing to be a VoxelSpacing instance")       

        if not isinstance(target_spacing, VoxelSpacing):
            raise TypeError("Expected target_spacing to be a VoxelSpacing instance")

        if volume.volume.ndim != 3:
            raise ValueError("MRI must be a 3 dimensional model")

        current = np.array(current_spacing.as_tuple, dtype=float)

        target = np.array(target_spacing.as_tuple, dtype=float)     

        if np.any(current <= 0):
            raise ValueError("Current voxel spacing values must be greater tha zero")

        if np.any(target <= 0):
            raise ValueError("Target voxel spacing values must be greater tha zero")

        # --------------------------------------------------------------------------
        # Calculate scaling factors
        #
        # Example:
        # current = 4 mm
        # target  = 2 mm
        #
        # zoom = 4 / 2 = 2
        #
        # Therefore the number of voxels doubles.
        # --------------------------------------------------------------------------

        zoom_factors = tuple(
            current_value / target_value
            for current_value, target_value 
            in zip(current, target)
        )

        # --------------------------------------------------------------------------
        # Resample the volume
        # --------------------------------------------------------------------------

        resampled_volume = zoom(
            volume.volume,
            zoom=zoom_factors,
            order=1
        )

        resampled_volume = np.ascontiguousarray(resampled_volume)

        return DicomVolume(
            study_instance_uid=volume.study_instance_uid,
            series_instance_uid=volume.series_instance_uid,
            volume=resampled_volume,
            source_path=volume.source_path
        )