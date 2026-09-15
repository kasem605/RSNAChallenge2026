from .dicom_series import DicomSeries
from .dicom_validation_result import DicomValidationResult
class DicomSeriesValidator:

    """
    Validates the structural integrity of a DICOM series.
    """

    def validate(self, series: DicomSeries) -> DicomValidationResult:

        errors: list[str] = []

        if not series.slices:
            errors.append("Series contains no DICOM slices.")

        image_shape = series.image_shape

        if image_shape is None:
            errors.append("Series contains no image data. ")

        for index, slice_ in enumerate(series.slices):

            if slice_.pixel_array is None:
               errors.append(f"Slice {index} contains no pixel data.")
               continue

            if image_shape is not None:
                if slice_.pixel_array.shape != image_shape:
                    errors.append(
                        f"Slice {index} has shape"
                        f"{slice_.pixel_array.shape}"
                        f"expected {image_shape}."
                    )

        return DicomValidationResult(
            is_valid=len(errors) == 0,
            image_count = len(series.slices),
            image_shape=image_shape,
            errors=tuple(errors)
        )
