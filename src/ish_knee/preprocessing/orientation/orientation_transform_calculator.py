
from dataclasses import dataclass
import numpy as np

from ..dicom.dicom_series import DicomSeries
from ..volume.orientation_transform import OrientationTransform


@dataclass(frozen=True)
class OrientationTransformCalculator:
    """
    Calculates a permutation and flips that align a DICOM
    volume with the requested patient-coordinate axis order.

    Target array axes:
        axis 0 -> patient Z (superior/inferior)
        axis 1 -> patient Y (anterior/posterior)
        axis 2 -> patient X (left/right)

    This supports axis-aligned or nearly axis-aligned series.
    It does not correct oblique image geometry.
    """

    minimum_alignment: float = 0.80

    def calculate(self, series: DicomSeries) -> OrientationTransform:
        if not series.slices:
            raise ValueError("Cannot calculate orientation for an empty series.")

        first_slice = series.ordered_slices()[0]
        orientation = first_slice.image_orientation

        if orientation is None or len(orientation) != 6:
            raise ValueError(
                "A valid ImageOrientationPatient is required."
            )

        # DICOM IOP:
        # first triplet: direction of increasing column index
        # second triplet: direction of increasing row index
        column_direction = np.asarray(orientation[:3], dtype=float)
        row_direction = np.asarray(orientation[3:], dtype=float)

        normal = np.cross(column_direction, row_direction)

        for name, vector in (
            ("column direction", column_direction),
            ("row direction", row_direction),
            ("slice normal", normal),
        ):
            length = np.linalg.norm(vector)
            if not np.isfinite(length) or length < 1e-8:
                raise ValueError(f"Invalid {name} vector.")

        column_direction /= np.linalg.norm(column_direction)
        row_direction /= np.linalg.norm(row_direction)
        normal /= np.linalg.norm(normal)

        # Current NumPy axes map to patient-coordinate directions.
        # Slice ordering is increasing along the normal because
        # DicomSeries.ordered_slices() sorts by projected position.
        source_directions = np.stack(
            [normal, row_direction, column_direction],
            axis=0,
        )

        # Target array axes are Z, Y, X.
        target_directions = np.eye(3)[[2, 1, 0]]

        # Match each target axis to the best unused source axis.
        # Evaluate all six permutations and select the strongest match.
        import itertools

        best_order = None
        best_score = -1.0
        best_signs = None

        for order in itertools.permutations((0, 1, 2)):
            signs = []
            scores = []

            for target_axis, source_axis in enumerate(order):
                direction = source_directions[source_axis]
                target = target_directions[target_axis]

                dot = float(np.dot(direction, target))
                signs.append(dot < 0)
                scores.append(abs(dot))

            score = float(np.mean(scores))

            if score > best_score:
                best_score = score
                best_order = order
                best_signs = signs

        if best_score < self.minimum_alignment:
            raise ValueError(
                f"Series is too oblique for a simple permutation/flip: "
                f"alignment score={best_score:.3f}"
            )

        return OrientationTransform(
            axis_order=tuple(best_order),
            flip_axis_0=bool(best_signs[0]),
            flip_axis_1=bool(best_signs[1]),
            flip_axis_2=bool(best_signs[2]),
        )
