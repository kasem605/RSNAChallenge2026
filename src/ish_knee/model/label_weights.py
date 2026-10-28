from __future__ import annotations

import torch

from ish_knee.dataset.dataset_sample import DatasetSample


LABEL_ATTRIBUTES = (
    "acl",
    "mcl",
    "medial_meniscus",
    "lateral_meniscus",
    "medial_oa",
    "lateral_oa",
    "pf_oa",
    "effusion",
    "synovitis",
    "bakers",
    "contusion",
    "fracture",
)


def calculate_positive_weights(
    training_samples: list[DatasetSample],
) -> list[float]:
    """
    Calculate one positive-class weight per label.

    Formula:
        positive weight = negative count / positive count

    Uses training samples only. Validation data must not be used
    to calculate training weights.
    """

    if not training_samples:
        raise ValueError("training_samples cannot be empty.")

    weights: list[float] = []

    for label_name in LABEL_ATTRIBUTES:
        positive_count = sum(
            int(getattr(sample.labels, label_name))
            for sample in training_samples
        )

        negative_count = len(training_samples) - positive_count

        if positive_count == 0:
            raise ValueError(
                f"Cannot calculate a positive weight for '{label_name}': "
                "there are no positive training examples."
            )

        if negative_count == 0:
            raise ValueError(
                f"Cannot calculate a positive weight for '{label_name}': "
                "there are no negative training examples."
            )

        weight = negative_count / positive_count
        weights.append(float(weight))

    if len(weights) != 12 or not torch.isfinite(
        torch.tensor(weights, dtype=torch.float32)
    ).all():
        raise ValueError("Calculated positive weights are invalid.")

    return weights
