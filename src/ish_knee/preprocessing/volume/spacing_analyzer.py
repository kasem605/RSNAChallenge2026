import numpy as np
from ..dicom.dicom_series import DicomSeries
from .voxel_spacing import VoxelSpacing

class SpacingAnalyzer:

    """
    Determines the physical voxel spacing of a DICOM series

    Spacing is returned in millimeters

    Axis order:

    axis 0 = slice / depth
    axis 1 = row
    axis 2 = column
    """

    def analyze(self, series: DicomSeries) -> VoxelSpacing:

        if not isinstance(series, DicomSeries):
            raise TypeError("Expected a DicomSeries instance")

        if not series.slices:
            raise ValueError("Cannot determine spacing from an empty DICOM series")

        first_slice = series.slices[0]

        # ---------------------------------------------------------------
        # Read PixelSpacing
        # ---------------------------------------------------------------

        dataset = self._read_metadata(
            first_slice.path
        )

        pixel_spacing = getattr(
            dataset,
            "PixelSpacing",
            None
        )

        if pixel_spacing is None:
            raise ValueError("DICOM series does not contain PixelSpacing.")

        if len(pixel_spacing) < 2:
            raise ValueError("DICOM PixelSpacing must contain at least two values.")

        row_spacing = float(pixel_spacing[0])

        column_spacing = float(pixel_spacing[1])

        if row_spacing <= 0:
            raise ValueError("Row pixel spacing must be greater than zero")

        if column_spacing <= 0:
            raise ValueError("Column pixel spacing must be greater than zero")

        # ----------------------------------------------------------------
        # Determine slice spacing
        # ----------------------------------------------------------------

        slice_spacing = self._calculate_slice_spacing(series)

        return VoxelSpacing(
            spacing_axis_0=slice_spacing,
            spacing_axis_1=row_spacing,
            spacing_axis_2=column_spacing
        )
    
    @staticmethod
    def _read_metadata(path):

        import pydicom

        return pydicom.dcmread(
            path,
            stop_before_pixels=True,
            force=True
        )
    
    @staticmethod
    def _calculate_slice_spacing(series: DicomSeries) -> float:

        ordered_slices = series.ordered_slices()

        if len(ordered_slices) < 2:
            raise ValueError("At least two slices are required to determine slice spacing")

        # -------------------------------------------------------------------
        # Get the image orientation from the first slice
        # -------------------------------------------------------------------

        orientation = ordered_slices[0].image_orientation

        if orientation is None:
            raise ValueError("Orientation is required to determine slice spacing.")

        if len(orientation) != 6:
            raise ValueError("Image orientation must contain exactly 6 values.")

        # -------------------------------------------------------------------
        # Calculate the size normal
        # -------------------------------------------------------------------

        row_direction = np.array(orientation[0:3], dtype=float)

        column_direction = np.array(orientation[3:6], dtype=float)

        slice_normal = np.cross(row_direction, column_direction)

        normal_length = np.linalg.norm(slice_normal)

        if normal_length == 0:
            raise ValueError("Unable to calculate a valid slice normal")

        slice_normal =(slice_normal / normal_length)

        # -------------------------------------------------------------------
        # Project each slice position onto the slice normal
        # -------------------------------------------------------------------        

        projected_positions = []

        for slice_ in ordered_slices:

            if slice_.image_position is None:
                continue

            position = np.asarray(
                    slice_.image_position,
                    dtype=float
                )

            projected_position = np.dot(position, slice_normal)

            projected_positions.append(float(projected_position))

        if len(projected_positions) < 2:
            raise ValueError("At least two valid slice positions are required to determine slice spacing")

        # -------------------------------------------------------------------
        # Calculate adjusted slice spacing
        # -------------------------------------------------------------------

        distances = []

        for first, second in zip(
            projected_positions[:-1],
            projected_positions[1:]
        ):
            distance = abs(second - first)

            if distance > 0:
                distances.append(distance)

        if not distances:
            raise ValueError("Unable to determine slice spacing.")

        # -------------------------------------------------------------------
        # Median provides robustness against an occasional irregular position.
        # -------------------------------------------------------------------

        return float(np.median(distances))

