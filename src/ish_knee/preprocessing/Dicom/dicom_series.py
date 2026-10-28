
from dataclasses import dataclass

import numpy as np

from .dicom_slice import DicomSlice


@dataclass(frozen=True)
class DicomSeries:
    """
    Represents a collection of DICOM slices belonging to one MRI series.
    """

    study_instance_uid: str
    series_instance_uid: str
    slices: tuple[DicomSlice, ...]

    @property
    def image_count(self) -> int:
        """Return the number of DICOM images in the series."""
        return len(self.slices)

    @property
    def image_shape(self) -> tuple[int, int] | None:
        """
        Return the first slice's (rows, columns) shape,
        or None if the series has no usable first slice.
        """
        if not self.slices:
            return None

        pixel_array = self.slices[0].pixel_array

        if pixel_array is None:
            return None

        return pixel_array.shape

    def ordered_slices(self) -> tuple[DicomSlice, ...]:
        """
        Return slices in spatial order when valid geometry is available.

        Uses ImagePositionPatient projected onto the slice normal.
        Falls back to InstanceNumber when geometry is unavailable
        or inconsistent.

        Note:
            InstanceNumber is not guaranteed to represent physical order.
        """
        if not self.slices:
            return ()

        # Spatial sorting requires position and orientation for every slice.
        has_geometry = all(
            slice_.image_position is not None
            and slice_.image_orientation is not None
            for slice_ in self.slices
        )

        if not has_geometry:
            return self._order_by_instance_number()

        try:
            # Read and validate the first slice's orientation.
            orientation = np.asarray(
                self.slices[0].image_orientation,
                dtype=np.float64,
            )

            if orientation.shape != (6,):
                return self._order_by_instance_number()

            if not np.all(np.isfinite(orientation)):
                return self._order_by_instance_number()

            row_direction = orientation[:3]
            column_direction = orientation[3:]

            # DICOM direction vectors should be unit length and orthogonal.
            if not np.isclose(
                np.linalg.norm(row_direction), 1.0, atol=1e-3
            ):
                return self._order_by_instance_number()

            if not np.isclose(
                np.linalg.norm(column_direction), 1.0, atol=1e-3
            ):
                return self._order_by_instance_number()

            if not np.isclose(
                np.dot(row_direction, column_direction),
                0.0,
                atol=1e-3,
            ):
                return self._order_by_instance_number()

            # The normal is perpendicular to the image plane.
            slice_normal = np.cross(
                row_direction,
                column_direction,
            )

            normal_length = np.linalg.norm(slice_normal)

            if not np.isfinite(normal_length) or normal_length < 1e-8:
                return self._order_by_instance_number()

            slice_normal = slice_normal / normal_length

            # Confirm that every slice has a compatible orientation.
            for slice_ in self.slices:
                current_orientation = np.asarray(
                    slice_.image_orientation,
                    dtype=np.float64,
                )

                if current_orientation.shape != (6,):
                    return self._order_by_instance_number()

                if not np.all(np.isfinite(current_orientation)):
                    return self._order_by_instance_number()

                if not np.allclose(
                    current_orientation,
                    orientation,
                    atol=1e-3,
                    rtol=0.0,
                ):
                    return self._order_by_instance_number()

            # Project each slice's patient-space position onto the normal.
            def slice_location(slice_: DicomSlice) -> float:
                position = np.asarray(
                    slice_.image_position,
                    dtype=np.float64,
                )

                if position.shape != (3,):
                    raise ValueError(
                        "ImagePositionPatient must contain three values."
                    )

                if not np.all(np.isfinite(position)):
                    raise ValueError(
                        "ImagePositionPatient contains invalid values."
                    )

                return float(np.dot(position, slice_normal))

            locations = [
                slice_location(slice_)
                for slice_ in self.slices
            ]

            if not np.all(np.isfinite(locations)):
                return self._order_by_instance_number()

            return tuple(
                slice_
                for _, slice_ in sorted(
                    zip(locations, self.slices),
                    key=lambda item: item[0],
                )
            )

        except (TypeError, ValueError, IndexError):
            return self._order_by_instance_number()

    def _order_by_instance_number(self) -> tuple[DicomSlice, ...]:
        """Fallback ordering when spatial geometry cannot be used."""
        return tuple(
            sorted(
                self.slices,
                key=lambda slice_: slice_.instance_number,
            )
        )

    