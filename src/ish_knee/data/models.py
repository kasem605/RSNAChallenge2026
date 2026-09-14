
from dataclasses import dataclass
from typing import Optional, Tuple

# Describes a series, not an individual slice

@dataclass
class SeriesMetadata:
    study_instance_uid: str
    series_instance_uid: str
    series_number: Optional[int] = None
    series_description: Optional[str] = None
    protocol_name: Optional[str] = None
    sequence_name: Optional[str] = None

    modality: Optional[str] = None
    body_part_examined: Optional[str] = None

    manufacturer: Optional[str] = None
    manufacturer_model_name: Optional[str] = None

    magnetic_field_strength: Optional[float] = None

    rows: Optional[int] = None
    columns: Optional[int] = None

    pixel_spacing: Optional[Tuple[float, float]] = None

    slice_thickness: Optional[float] = None
    spacing_between_slices: Optional[float] = None

    image_orientation_patient: Optional[Tuple[float, ...]] = None
    image_position_patient: Optional[Tuple[float, ...]] = None

    photometric_interpretation: Optional[str] = None

    rescale_slope: Optional[float] = None
    rescale_intercept: Optional[float] = None

    number_of_frames: Optional[int] = None

    file_count: int = 0

    inferred_plane: Optional[str] = None
