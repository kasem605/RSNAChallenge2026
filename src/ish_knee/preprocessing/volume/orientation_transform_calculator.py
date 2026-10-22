
from dataclasses import dataclass
import numpy as np

from .orientation_info import OrientationInfo
from .orientation_transform import OrientationTransform


@dataclass
class OrientationTransformCalculator:
    """
    Calculates a permutation and axis flips to align a volume
    with canonical patient-coordinate axes.

    Assumes source volume axes are:
        axis 0: slice / depth
        axis 1: pixel-array row
        axis 2: pixel-array column

    Canonical output axes correspond to patient Z, Y, X.
    """

    def calculate(
        self,
        orientation: OrientationInfo,
    ) -> OrientationTransform:

        if not isinstance(orientation, OrientationInfo):
            raise TypeError("Expected an OrientationInfo instance.")

        row = np.asarray(orientation.row_direction, dtype=float)
        column = np.asarray(orientation.column_direction, dtype=float)
        normal = np.asarray(orientation.slice_normal, dtype=float)

        # DICOM pixel-array axis 1 follows the column-direction
        # vector; axis 2 follows the row-direction vector.
        source_directions = np.column_stack(
            (normal, row, column)
        )

        if not np.isfinite(source_directions).all():
            raise ValueError("Orientation vectors contain invalid values.")

        norms = np.linalg.norm(source_directions, axis=0)
        if np.any(norms < 1e-6):
            raise ValueError("Orientation contains a zero-length direction.")

        source_directions = source_directions / norms

        # Canonical output axes: superior/inferior, anterior/posterior,
        # left/right in DICOM patient coordinates.
        canonical_directions = np.eye(3)[:, [2, 1, 0]]

        # Each output axis selects one source axis. Find the best
        # one-to-one mapping by maximizing absolute alignment.
        scores = np.abs(canonical_directions.T @ source_directions)

        import itertools
        candidates = list(itertools.permutations(range(3)))
        axis_order = max(
            candidates,
            key=lambda order: sum(
                scores[out_axis, order[out_axis]]
                for out_axis in range(3)
            ),
        )

        alignment = [
            float(
                np.dot(
                    canonical_directions[:, out_axis],
                    source_directions[:, axis_order[out_axis]],
                )
            )
            for out_axis in range(3)
        ]

        # Reject strongly oblique orientations rather than silently
        # treating a substantial rotation as a simple flip/permutation.
        if any(abs(value) < 0.8 for value in alignment):
            raise ValueError(
                "Series is substantially oblique. "
                "A permutation/flip alone may not canonicalize it."
            )

        flips = tuple(value < 0 for value in alignment)

        return OrientationTransform(
            axis_order=tuple(axis_order),
            flip_axis_0=flips[0],
            flip_axis_1=flips[1],
            flip_axis_2=flips[2],
        )
