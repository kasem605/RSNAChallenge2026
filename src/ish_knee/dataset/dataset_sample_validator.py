from .dataset_sample import DatasetSample

class DatasetSampleValidator:

    """
    Validates a collection of DatasetSample objects

    A valid sample must contain:
        - a study UID
        - all three MRI planes
        - 12 abnormality labels
        - binary label values
    """

    EXPECTED_LABEL_COUNT = 12

    def validate(self, samples: list[DatasetSample]) -> None:

        if not isinstance(samples, list):
            raise TypeError("samples must be a list of DatasetSample objects")

        if len(samples) == 0:
            raise ValueError("No dataset samples were provided ")

        for index, sample in enumerate(samples):
            
            if not isinstance(sample, DatasetSample):
                raise TypeError(f"Sample at {index} is nota Dataset sample instance")

            if not sample.study.study_instance_uid:
                raise ValueError(f"Sample at index {index} has an empty StudyInstanceUID")

            if not sample.study.is_complete:
                raise ValueError(f"Sample {sample.study_instance_uid} does not contain all MRI planes")

            if sample.labels.count != self.EXPECTED_LABEL_COUNT:
                raise ValueError(f"Study {sample.study_instance_uid} has {sample.labels.count} labels. Expected {self.EXPECTED_LABEL_COUNT}")

            if any(
                label not in (0,1)
                for label in sample.labels.as_tuple
            ):
                raise ValueError(f"Study {sample.study_instance_uid} contains an invalid lable")

    def count_complete_samples(
                self,
                samples: list[DatasetSample]
        ) -> int:
            return sum(
                sample.study.is_complete
                for sample in samples
            )