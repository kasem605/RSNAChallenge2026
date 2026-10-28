from dataclasses import dataclass

@dataclass(frozen=True)
class VoxelSpacing:
    """
    Represents the physical spacing voxels
    in a 3-D MRI volume

    values are expressed in millimeters

    Axis order:
        axis 0 = depth / slice direction
        axis 1 = row direction
        axis 2 = column direction
    """

    spacing_axis_0: float
    spacing_axis_1: float
    spacing_axis_2: float

    @property
    def as_tuple(self) -> tuple[float, float, float]:
        """
        Return voxel spacing as a tuple
        """

        return(
            self.spacing_axis_0,
            self.spacing_axis_1,
            self.spacing_axis_2                       
        )