from dataclasses import dataclass

import numpy as np

from ..dataset.knee_mri_sample import KneeMRISample

@dataclass(frozen=True)
class ModelInput:

    """
    Represents the MRI data supplied to the model

    Each volume is a processed 3-D Numpy array

    Axis convention:
        axis 0 = depth /slice
        axis 1 = height / row
        axis 2 = width / column
    """

    study_instance_uid: str

    sagittal: np.ndarray
    coronal: np.ndarray
    axial: np.ndarray

    def __post_init__(self) -> None:

        if not self.study_instance_uid:
            raise ValueError("study_instance_uid cannot be empty")

        volumes = (
            ("sagittal",self.sagittal),
            ("coronal",self.coronal),
            ("axial",self.axial),
        )

        for name, volume in volumes:

            if not isinstance(volume, np.ndarray):
                raise TypeError(f"{name} volume must ne a numpy array")

            if volume.ndim != 3:
                raise ValueError(f"{name} volume must be 3-dimenaional")

    @classmethod
    def from_mri_sample(cls, mri_sample: KneeMRISample) -> "ModelInput":

        if not isinstance(mri_sample, KneeMRISample):
            raise TypeError("mri_sample must be a KneeMRIInstance instance") 

        return cls(
                study_instance_uid = mri_sample.study_instance_uid,
                sagittal = mri_sample.sagittal,
                coronal = mri_sample.coronal,
                axial = mri_sample.axial
        )      

        @property
        def sagittal_shape(self) -> tuple[int, ...]:
            return self.sagittal.shape

        @property
        def coronal_shape(self) -> tuple[int, ...]:
            return self.coronal.shape

        @property
        def axial_shape(self) -> tuple[int, ...]:
            return self.axial.shape
