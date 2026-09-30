from dataclasses import dataclass

from .knee_study_sample import KneeStudySample
from .knee_labels import KneeLabels

@dataclass(frozen=True)
class DatasetSample:

    """
    Represents one complete traiining sample.

    contains:
        - MRI study information
        - abnormality labels
    """

    study: KneeStudySample
    labels: KneeLabels

    @property
    def study_instance_uid(self) -> str:
        return self.study.study_instance_uid