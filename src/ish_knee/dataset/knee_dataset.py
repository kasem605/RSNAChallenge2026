from .dataset_sample import DatasetSample
from .knee_mri_sample import KneeMRISample
from .knee_mri_processor import KneeMRIProcessor

class KneeDataset:

    """
    Provides access to knee MRI dataset samples.

    MRI volumes are processed on demand rather than being preloaded into memory
    """

    def __init__(self, samples: list[DatasetSample], mri_processor: KneeMRIProcessor)-> None:

        if not isinstance(samples, list):
            raise TypeError("samples must be a list of DatasetSample objects.")

        for sample in samples:
            if not isinstance(sample, DatasetSample):
                raise TypeError("All samples must be aDatasetSample objects")

        if not isinstance(mri_processor, KneeMRIProcessor):
            raise TypeError("mri_processor must be a KneeMRIProcessor")

        if len(samples) == 0:
            raise ValueError("Atleast one dataset sample is required.")

        self._samples = tuple(samples)

        self._mri_processor = mri_processor

    def __len__(self) -> int:

        return len(self._samples)

    def get_sample(self, index: int) -> DatasetSample:

        """
        Returns the metadata and labels for one study
        """

        if not isinstance(index, int):
            raise TypeError(index must be an integer)

        if index < 0 or index >= len(self._samples):
            raise ValueError(f"Dataset index out of range: {index}")

        return self._samples[index]

    def get_mri(self, index: int) -> KneeMRISample:

        """
        Processes and returns the MRI volumes for one study
        """

        sample = self.get_sample(index)

        return self._mri_processor(sample)

    def get_study_uid(self, index: int) -> str:

        """
        Returns the StudyInstanceUID for one sample
        """

        return self.get_sample(index).study_instance_uid




