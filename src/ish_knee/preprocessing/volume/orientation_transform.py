from dataclasses import dataclass

@dataclass(frozen=True)
class OrientationTransform:

    """
    Describes a transformation required to convert a volume
    in the the canonical orientation
    """

    axis_order: tuple[int, int, int]

    flip_axis_0: bool
    flip_axis_1: bool
    flip_axis_2: bool

    @property
    def is_identity(self) -> bool:

        """
        Returns True when no axis permutation or flip is requiered.
        """

        return(
            self.axis_order== (0, 1, 2)
            and not self.flip_axis_0
            and not self.flip_axis_1
            and not self.flip_axis_2
        )