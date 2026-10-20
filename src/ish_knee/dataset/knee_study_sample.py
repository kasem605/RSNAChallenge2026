from dataclasses import dataclass

from ..preprocessing.series_selector import SelectedSeries

@dataclass(frozen=True)
class KneeStudySample:

    """
    Represents one RSNA knee MRI study.

    A study may contain up to three selected MRI planes:
        - sagittal
        - coronal
        - axial

    The MRI image data itself is not stored here.
    This class stores the selected series information
    needed to locate and preprocess the volumes
    """

    study_instance_uid: str

    sagittal: SelectedSeries | None
    coronal: SelectedSeries | None
    axial: SelectedSeries | None

    @property
    def has_sagittal(self) -> bool:
        return self.sagittal is not None

    @property
    def has_coronal(self) -> bool:
        return self.coronal is not None

    @property
    def has_axial(self) -> bool:
        return self.axial is not None    

    @property
    def plane_count(self) -> int:
        return sum(
            plane is not None
            for plane in (
                self.sagittal,
                self.coronal,
                self.axial
            )
        )

    @property
    def is_complete(self) -> bool:
        """
        Returns True when all three MRI planes are available.
        """

        return (
            self.sagittal is not None
            and self.coronal is not None
            and self.axial is not None
        )

    