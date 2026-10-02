from ..preprocessing.volume.orientation_info import OrientationInfo
from ..preprocessing.volume.orientation_transform import OrientationTransform

import numpy as np

class OrientationTransformBuilder:
    """
    Builds an OrientationTransform from DICOM orientation information.

    The builder determines how the current DICOM axes relate to the
    canonical volume axes.

    Canbonical volume convention:
        axis 0 = slice/depth
        axis 1 = row
        axis 2 = column
    """

    def build(self, orientation: OrientationInfo)-> OrientationTransform:

        if not isinstance(orientation, OrientationInfo):
            raise TypeError("Expected an OrientationInfo instance")

        row_direction = np.array(orientation.row_direction, dtype=float)

        column_direction = np.array(orientation.column_direction, dtype=float)

        slice_normal = np.array(orientation.slice_normal)

        axis_order = self._determine_axis_order(row_direction, column_direction, slice_normal)

        return OrientationTransform(
            axis_order=axis_order,
            flip_axis_0=False,
            flip_axis_1= False,
            flip_axis_2= False
        )

    @staticmethod
    def _determine_axis_order(row_direction: np.ndarray, 
                               column_direction: np.ndarray, 
                               slice_normal: np.ndarray) -> tuple[int, int, int]:

        directions = (slice_normal, row_direction, column_direction)

        dominant_axes = []

        for direction in directions:

            dominant_axis = int(np.argmax(np.abs(direction)))

            dominant_axes.append(dominant_axis)

        return tuple(dominant_axes)
        
    