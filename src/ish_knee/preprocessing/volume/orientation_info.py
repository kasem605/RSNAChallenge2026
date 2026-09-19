from dataclasses import dataclass

@dataclass(frozen=True)
class OrientationInfo:
    """
    Describes the geometric orientation of a Dicom series.
    """

    row_direction: tuple[float, float, float]
    column_direction: tuple[float, float, float]
    slice_normal: tuple[float, float, float]
    anatomical_plane: str