from pathlib import Path

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder
from ish_knee.dataset.labeled_dataset_builder import LabeledDatasetBuilder
from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.model.study_split import StudySplit

DATA_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")
VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42

def main() -> None:

    print("=" * 70)
    print("RSNA KNEE AI - TRAINING LABEL BALANCE")
    print("=" * 70)

    paths = DatasetPaths.from_root(DATA_ROOT)
    metadata = MetadataReader(paths)
    metadata.validate()

    sample_builder = DatasetSampleBuilder(metadata=metadata, series_selector=SeriesSelector(paths.train_series_dir))

    labeled_builder = LabeledDatasetBuilder(sample_builder)
    samples = labeled_builder.build_all()

    training_samples, validation_samples = StudySplit.split(
        samples,
        validation_fraction = VALIDATION_FRACTION,
        seed = RANDOM_SEED
    )

    print(f"\n Fully labeled studies: {len(samples)}")
    print(f"Training studies: {len(training_samples)}")
    print(f"Validation studies:     {len(validation_samples)}")

    if not training_samples:
        raise RuntimeError("This training split is empty")

    label_names = [
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

    positive_counts =[0] * len(label_names)

    for sample in training_samples:
        values = sample.labels.as_tuple

        if len(values) != len(label_names):
            raise ValueError(f"Expected {len(label_names)} labels, got {len(values)}")

        for index, value in enumerate(values):
            if value not in (0,1):
                raise ValueError(f"Unexpected label value {value!r} for {label_names[index]}")

            positive_counts[index] += int(value)

    print("\nTRAINING SPLIT - PER-LABEL COUNTS")
    print("-" * 70)
    print(
        f"{'Label':<22}"
        f"{'Positive':>10}"
        f"{'Negative':>10}"
        f"{'Positive %':>13}"
        f"{'Neg/Pos':<22}"
    )

    for label, positives in zip(label_names, positive_counts):
        negatives = len(training_samples) - positives
        positive_pct = 100.0 * positives / len(training_samples)

        ratio = (
            f"{negatives / positives:.2f}"
            if positives > 0
            else "N/A"
        )
        print(
            f"{label:<22}"
            f"{positives:>10}"
            f"{negatives:>10}"
            f"{positive_pct:>12.1f}"
            f"{ratio:>10}"
        )

    print("\nInterpretetion")
    print("- Counts use trainingstudies only. ")  
    print("- Validation labelsare not used to calculate these counts")    
    print("- Very small positive counts can make class weighting unstable.")
    print("\nLABEL BALANCE ANALYSIS PASSED")


if __name__ == "__main__":
        main()