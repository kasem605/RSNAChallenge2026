from dataclasses import dataclass

from .dicom_series import DicomSeries

@dataclass(frozen=True)
class DicomValidationResult:

    """
    Result of validating a DICOM series
    """

    is_valid: bool
    image_count: int
    image_shape: tuple[int, int] | None
    errors: tuple[str, ...]

    
