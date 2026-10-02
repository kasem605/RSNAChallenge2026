from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class KneeMRISample:

    """
    Represent the processed MRI volumes for one study.

    Each volume is expected to be a 3-D Numpy array

    axis convention:
        axis 0 = depth / slice
        axis 1 = height / row
        axis 2 = width /column
    """

    study_instance_uid: str

    sagittal: np.ndarray
    coronal: np.ndarray
    axial: np.ndarray

    @property
    def sagittal_shape(self) -> tuple[int, ...]:
        return self.sagittal.shape

    @property
    def coronal_shape(self) -> tuple[int, ...]:
        return self.coronal.shape

    @property
    def axial_shape(self) -> tuple[int, ...]:
        return self.axial.shape

    @property
    def is_3d(self) -> bool:
        return (
            self.sagittal.ndim == 3
            and self.coronal.ndim == 3
            and self.axial.ndim ==3
        )

