
from pathlib import Path

import pandas as pd

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder
from ish_knee.dataset.labeled_dataset_builder import LabeledDatasetBuilder
from ish_knee.model.study_split import StudySplit


# Update this only if your dataset root is different.
DATA_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")

VALIDATION_FRACTION = 0.30
RANDOM_SEED = 42

LABEL_COLUMNS = LabeledDatasetBuilder.LABELED_COLUMNS


def print_distribution(
    title: str,
    dataframe: pd.DataFrame,
) -> None:
    """Print positive and negative counts for each label."""

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)

    print(
        f"{'Label':<22}"
        f"{'Positive':>10}"
        f"{'Negative':>10}"
        f"{'Unknown':>10}"
        f"{'Total':>10}"
        f"{'Positive %':>13}"
    )
    print("-" * 90)

    for label in LABEL_COLUMNS:
        values = dataframe[label]

        positive = int((values == 1).sum())
        negative = int((values == 0).sum())
        unknown = int(values.isna().sum())
        total = len(values)

        known = positive + negative
        positive_pct = (
            100.0 * positive / known
            if known > 0
            else 0.0
        )

        print(
            f"{label:<22}"
            f"{positive:>10}"
            f"{negative:>10}"
            f"{unknown:>10}"
            f"{total:>10}"
            f"{positive_pct:>12.1f}%"
        )

def samples_to_dataframe(sample_list):
    """Convert DatasetSample objects into a DataFrame of labels."""

    # Map original CSV column names to KneeLabels attributes.
    label_attribute_map = {
        "ACL": "acl",
        "MCL": "mcl",
        "Medial Meniscus": "medial_meniscus",
        "Lateral Meniscus": "lateral_meniscus",
        "Medial OA": "medial_oa",
        "Lateral OA": "lateral_oa",
        "PF OA": "pf_oa",
        "Effusion": "effusion",
        "Synovitis": "synovitis",
        "Baker's": "bakers",
        "Contusion": "contusion",
        "Fracture": "fracture",
    }

    records = []

    for sample in sample_list:
        labels = sample.labels

        record = {
            "StudyInstanceUID": sample.study_instance_uid,
        }

        for column_name, attribute_name in label_attribute_map.items():
            record[column_name] = getattr(labels, attribute_name)

        records.append(record)

    return pd.DataFrame(
        records,
        columns=["StudyInstanceUID", *LABEL_COLUMNS],
    )

def main() -> None:
    print("Loading dataset metadata...")

    paths = DatasetPaths.from_root(DATA_ROOT)
    metadata = MetadataReader(paths)
    metadata.validate()

    series_selector = SeriesSelector(paths.train_series_dir)
    sample_builder = DatasetSampleBuilder(
        metadata,
        series_selector,
    )
    labeled_builder = LabeledDatasetBuilder(sample_builder)

    # Builds samples only for studies with all 12 labels present.
    samples = labeled_builder.build_all()

    if not samples:
        raise ValueError("No fully labeled studies were found.")

    print(f"Fully labeled studies: {len(samples)}")

    # Reproduce the established experiment split.
    training_samples, validation_samples = StudySplit.split(
        samples,
        validation_fraction=VALIDATION_FRACTION,
        seed=RANDOM_SEED,
    )

    print(f"Training studies:      {len(training_samples)}")
    print(f"Validation studies:    {len(validation_samples)}")

    # Use the labels stored in each DatasetSample.
    # Convert them to a DataFrame without loading any MRI volumes.

    training_df = samples_to_dataframe(training_samples)
    validation_df = samples_to_dataframe(validation_samples)

    print_distribution("TRAINING LABEL DISTRIBUTION", training_df)
    print_distribution("VALIDATION LABEL DISTRIBUTION", validation_df)

    print()
    print("=" * 90)
    print("ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
