from dataclasses import dataclass

from .voxel_spacing import VoxelSpacing

@dataclass(frozen=True)
class ProcessingConfig:

    """
    Defines the configuration used b y the MRI
    volume preprocessing pipeline.
    """

    target_spacing: VoxelSpacing
    target_shape: tuple[int, int, int]

    normal_intensity: bool = True

    padding_value: float = 0.0

    def __post_init__(self) -> None:

        if not isinstance(self.target_spacing, VoxelSpacing):
            raise TypeError("target_spacing must be a VoxelSpacing instance.")

        if len(self.target_shape) != 3:
            raise ValueError("target_spacing must contain exactly three dimensions.")

        if any(
            dimension <= 0
            for dimension in self.target_shape
        ):
            raise ValueError("target_shape dimensions must be greater than zero.")