from .dataset_sample import DatasetSample
from .dataset_sample_builder import DatasetSampleBuilder

class LabeledDatasetBuilder:

    """
    Builds DatasetSample objects for studies that contain
    complete abnormality labels.
    """

    def __init__(self, sample_builder: DatasetSampleBuilder)->None:

        if not isinstance(sample_builder, DatasetSampleBuilder):
            raise TypeError("Sample builder must be a DataSampleBuilder instance")

        self._sample_builder = sample_builder

    def build_all(self) -> list[DatasetSample]:

        metadata = self._sample_builder._metadata

        train = metadata.train

        labeled_rows = train[train[self._sample_builder.LABELED_COLUMNS].notna().all(axis=1)]

        samples: list[DatasetSample] = []

        for study_instance_uid in labeled_rows["StudyInstanceUID"]:
            sample = self._sample_builder.build(study_instance_uid)
            samples.append(sample)

        return samples

    

