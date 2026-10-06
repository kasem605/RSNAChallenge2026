from .model_sample import ModelSample
from .model_sample_builder import ModelSampleBuilder

from ..dataset.dataset_sample import DatasetSample

class ModelDataset:

    """
    Provides model samples on demand

    Dataset samples are converted to ModelSample objects only
    when requested, avoiding loading all MRI volumes into memory
    """

    def __init__(self, samples: list[DatasetSample], sample_builder: ModelSampleBuilder)-> None:

        if not isinstance(samples, list):
            raise TypeError("samples must be a list of DatasetSample objects")

        for sample in samples:
            if not isinstance(sample, DatasetSample):
                raise TypeError("all samples must be DatsetSample objects")

        if not isinstance(sample_builder, ModelSampleBuilder):
            raise TypeError("sample_builder must be a ModelSampleBuilder instance")

        if len(samples) == 0:
            raise ValueError("at least one dataset sample is required")

        self._samples = tuple(samples)
        self._sample_builder = sample_builder

    def __len__(self) -> int:
        return len(self._samples)

    def get_sample(self, index: int) -> DatasetSample:
        if not isinstance(index, int):
            raise TypeError("index must be an integer")

        if index < 0 or index >= len(self._samples):
            raise IndexError(f"Dataset index out of range: {index}")

        return self._samples[index]

    def get_modal_sample(self, index: int) -> ModelSample:
        sample = self.get_sample(index)

        return self._sample_builder(index)

    def get_study_uid(self, index: int) -> str:
        return self.get_sample(index).study_instance_uid
