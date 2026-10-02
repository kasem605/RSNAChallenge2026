import random

from .dataset_sample import DatasetSample
from .dataset_split import DatasetSplit

class DatasetSplitter:

    """
    Splits labled knee MRI studies into training
    a dnvalidation datasets.

    Splitting is performed at the study level
    """

    def split(
            self,
            samples: list[DatasetSample],
            validation_fraction: float =0.20,
            random_seed: int = 42
    ) -> DatasetSplit:

        if not isinstance(samples, list):
            raise TypeError("Samples must be a list of DatasetSample objects")

        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")

        for sample in samples:
            if not isinstance(sample, DatasetSample):
                raise TypeError("All samples must be a DatasetSample objects")

        study_uids = [
            sample.study_instance_uid
            for sample in samples
        ]

        if len(study_uids) != len(set(study_uids)):
            raise ValueError("Duplicate StudyINstanceUID values detected")

        shuffled_samples = list(samples)

        random_generator = random.Random(random_seed)

        random_generator.shuffle(shuffled_samples)

        validation_count = max(1, int(len(shuffled_samples) * validation_fraction))

        validation_samples = tuple(shuffled_samples[:validation_count])

        train_samples = tuple(shuffled_samples[validation_count:])

        return DatasetSplit(
            train_samples=train_samples,
            validation_samples=validation_samples
        )
        
