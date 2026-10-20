from dataclasses import dataclass

import numpy as np

from ..dataset.knee_labels import KneeLabels

@dataclass(frozen=True)
class ModelTarget:

    """
    Represents the 12 abnormality targets supplied to the model
    """

    values: np.ndarray

    EXPECTED_COUNT = 12

    def __post_init__(self) -> None:

        if not isinstance(self.values,np.ndarray):
            raise TypeError("values must be numpy array")

        if self.values.ndim != 1:
            raise ValueError("Modeltarget values must be a 1-dimensional array")

        if len(self.values) != self.EXPECTED_COUNT:
            raise ValueError(f"ModelTarget must contain exactly {self.EXPECTED_COUNT} values")

        if not np.isin(self.values, [0,1]).all():
            raise ValueError("ModelTarget values must contain 0 or 1")

    @classmethod
    def from_labels(cls, labels: KneeLabels)-> "ModelTarget":

        if not isinstance(labels, KneeLabels):
            raise TypeError("labels must be a KneeLabels instance")

        values = np.asarray(labels.as_tuple, dtype=np.float32)

        return cls(values=values)

    @property
    def count(self) -> int:
        return len(self.values)
