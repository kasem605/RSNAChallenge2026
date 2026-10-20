from dataclasses import dataclass
from .dataset_sample import DatasetSample

@dataclass(frozen=True)
class DatasetSplit:
    """
    Contains training and validation samples.

    splitting is performed at the study level so that
    samples from the same study cannot appear in both datasets
    """

    train_samples: tuple[DatasetSample, ...]
    validation_samples: tuple[DatasetSample, ...]

    @property
    def train_count(self) -> int:
        return len(self.train_samples)

    @property
    def validation_count(self) -> int:
        return len(self.validation_samples)

    @property
    def total_count(self) -> int:
        return self.train_count + self.validation_count