import random
from typing import Sequence, TypeVar

class StudySplit:

    """
    Splits the study into training and validation groups.
    """

    T = TypeVar("T")

    @staticmethod
    def split(
        studies:Sequence[T],
        validation_fraction: float = 0.20,
        seed: int = 42
    ) -> tuple[list[T], list[T]]:

        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")

        if len(studies) < 2:
            raise ValueError("At least teo studies are required for splitting.")

        shuffled_studies = list(studies)
        random.Random(seed).shuffle(shuffled_studies)

        validation_count = round(len(shuffled_studies) * validation_fraction)

        validation_count = max(1, min(validation_count, len(shuffled_studies)))

        validation_studies = shuffled_studies[:validation_count]
        training_studies = shuffled_studies[validation_count:]

        return training_studies, validation_studies

