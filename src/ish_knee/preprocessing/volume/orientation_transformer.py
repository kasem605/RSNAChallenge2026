from dataclasses import dataclass
import numpy as np

from ..dicom.dicom_volume import DicomVolume
from .orientation_transform import OrientationTransform

@dataclass
class OrientationTransformer:

    """
    Applies an OrientationTransform to a DICOM volume.

    Axis permutation is applied first.
    Axis flips are then applied to the resulting volume
    """

    def transform(
            self,
            volume: DicomVolume,
            transformation: OrientationTransform
    ) -> DicomVolume:

        if not isinstance(volume, DicomVolume):
            raise TypeError("Expected a DicomVolume instance.")

        if not isinstance(transformation, OrientationTransform):
            raise TypeError("Expected an OrientationTransform instance.")

        if volume.volume.ndim != 3:
            raise ValueError("MRI volume must be 3-dimensional.")

        transformed_volume = np.array(
            volume.volume,
            copy=True
        )


        # ----------------------------------------------------------------
        # Step 1: Permute the axis
        # ----------------------------------------------------------------

        transformed_volume = np.transpose(
            volume.volume,
            axes = transformation.axis_order
        )

        # ----------------------------------------------------------------
        # Step 2: Apply flips
        # ----------------------------------------------------------------

        if transformation.flip_axis_0:
            transformed_volume = np.flip(
                transformed_volume,
                axis =0
            )

        if transformation.flip_axis_1:
            transformed_volume = np.flip(
                transformed_volume,
                axis =1
            )       

        if transformation.flip_axis_2:
            transformed_volume = np.flip(
                transformed_volume,
                axis =2
            ) 

        #  Make the resulting array contiguous in memory
        transformed_volume = np.ascontiguousarray(
            transformed_volume
        )
        return DicomVolume(
            study_instance_uid=volume.study_instance_uid,
            series_instance_uid=volume.series_instance_uid,
            volume=transformed_volume,
            source_path=volume.source_path
        )
