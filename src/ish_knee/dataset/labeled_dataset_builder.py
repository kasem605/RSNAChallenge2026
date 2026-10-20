from .dataset_sample import DatasetSample
from .dataset_sample_builder import DatasetSampleBuilder

class LabeledDatasetBuilder:

    """
    Builds DatasetSample objects for studies that contain
    complete abnormality labels.
    """

    LABELED_COLUMNS = [
            "ACL",
            "MCL",
            "Medial Meniscus",
            "Lateral Meniscus",
            "Medial OA",
            "Lateral OA",
            "PF OA",
            "Effusion",
            "Synovitis",
            "Baker's",
            "Contusion",
            "Fracture"
        ]
    
    def __init__(self, sample_builder: DatasetSampleBuilder)->None:

        if not isinstance(sample_builder, DatasetSampleBuilder):
            raise TypeError("Sample builder must be a DatasetSampleBuilder instance")

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

    def build_first(self) -> DatasetSample:

        """
        Builds the first study that contains complete labels
        """

        metadata = self._sample_builder._metadata
        train = metadata.train

        labeled_rows = train[train[list(self.LABELED_COLUMNS)].notna().all(axis=1)]

        if labeled_rows.empty:
            raise ValueError("No fully labeled studies were found")

        study_instance_uid = labeled_rows.iloc[0]["StudyInstanceUID"]

        return self._sample_builder.build(study_instance_uid)

