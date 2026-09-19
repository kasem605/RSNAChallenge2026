from .dicom_series import DicomSeries
from .dicom_validation_result import DicomValidationResult

import numpy as np

class DicomSeriesValidator:

    """
    Validates the structural integrity of a DICOM series.
    """

    def validate(self, series: DicomSeries) -> DicomValidationResult:

        errors: list[str] = []

        # ------------------------------------------------------------------------------------
        # Basic series validation
        # -----------------------------------------------------------------------------------

        if not series.slices:
            errors.append(
                "Series contains no DICOM slices."
                )
            return DicomValidationResult(
                is_valid=False,
                image_count=0,
                image_shape=None,
                errors=tuple(errors)
            )
        
        image_shape = series.image_shape

        if image_shape is None:
            errors.append("Series contains no image data. ")

        # ------------------------------------------------------------------------------------
        # Validate individual slices
        # -----------------------------------------------------------------------------------

        for index, slice_ in enumerate(series.slices):

            if slice_.pixel_array is None:
               errors.append(f"Slice {index} contains no pixel data.")
               continue

            if (image_shape is not None
                and slice_.pixel_array.shape != image_shape):
                    errors.append(
                        f"Slice {index} has shape "
                        f"{slice_.pixel_array.shape}"
                        f" expected {image_shape}."
                    )

        # ------------------------------------------------------------------------------------
        # Validate DICOM geometry
        # -----------------------------------------------------------------------------------

        missing_position = [
            index
            for index, slice_ in enumerate(series.slices)
            if slice_.image_position is None
        ]

        if missing_position:
            errors.append(
                f"{len(missing_position)} slice(s) "
                "are missing ImagePositionPatient"
            )

        missing_orientation = [
          index
            for index, slice_ in enumerate(series.slices)
            if slice_.image_orientation is None
        ]

        if missing_orientation:
            errors.append(
                f"{len(missing_orientation)} slice(s) "
                "are missing ImageOrientationPatient"
            )

        # ------------------------------------------------------------------------------------
        # Validate orientation consistency
        # -----------------------------------------------------------------------------------

        orientations = [
            slice_.image_orientation
            for slice_ in series.slices
            if slice_.image_orientation is not None
        ]

        if orientations:

            reference_orientation = np.array(
                orientations[0],
                dtype=float
            )

            for index, orientation in enumerate(
                orientations[1:],
                start=1
            ):
                current_orientation = np.array(
                    orientation,
                    dtype=float
                )

                if not np.allclose(
                    current_orientation,
                    reference_orientation,
                    atol = 1e-4
                ):
                    errors.append(
                        "ImageOrientationPatient is "
                        f"inconsistent at slice {index}"
                    )

        # ------------------------------------------------------------------------------------
        # Validate spatial ordering
        # -----------------------------------------------------------------------------------

        ordered_slices = series.ordered_slices()

        if len(ordered_slices) > 1:

            positions = [
                slice_.image_position
                for slice_ in ordered_slices
                if slice_.image_position is not None
            ]

            orientation = (ordered_slices[0].image_orientation)

            if(
                len(positions) == len(ordered_slices)
                and orientation is not None
            ):
                row_orientation = np.array(
                    orientation[0:3],
                    dtype=float
                )

                column_orientation = np.array(
                    orientation[3:6],
                    dtype=float
                )

                slice_normal = np.cross(
                    row_orientation,
                    column_orientation
                )

                locations = [
                    float(
                        np.dot(
                            np.array(position, dtype=float),
                            slice_normal
                        )
                    )
                    for position in positions
                ]

                # -------------------------------------------------------------
                # Check monotonic ordering
                # -------------------------------------------------------------

                differences = np.diff(locations)

                if np.any(differences <= 0):
                    errors.append(
                        "Slices are not in strictly increasing "
                        "spatial order."
                    )

                # -------------------------------------------------------------
                # Check duplicate positions
                # -------------------------------------------------------------

                if len(locations) != len(
                    set(
                        round(location, 5)
                        for location in locations
                    )
                ):
                    errors.append(
                        "Duplicate slice positions detected."
                    )

        return DicomValidationResult(
            is_valid=len(errors) == 0,
            image_count=series.image_count,
            image_shape=image_shape,
            errors=tuple(errors)
        )
