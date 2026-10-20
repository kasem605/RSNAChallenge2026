import json
from pathlib import Path

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder
from ish_knee.dataset.labeled_dataset_builder import LabeledDatasetBuilder
from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.model.study_split import StudySplit


DATA_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "evaluation"
OUTPUT_PATH = OUTPUT_DIR / "class_weights.json"

VALIDATION_FRACTION = 0.30
RANDOM_SEED = 42

LABEL_NAMES = [
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
    "Fracture",
]


def main() -> None:
    paths = DatasetPaths.from_root(DATA_ROOT)
    metadata = MetadataReader(paths)
    metadata.validate()

    sample_builder = DatasetSampleBuilder(
        metadata=metadata,
        series_selector=SeriesSelector(paths.train_series_dir),
    )
    samples = LabeledDatasetBuilder(sample_builder).build_all()

    training_samples, validation_samples = StudySplit.split(
        samples,
        validation_fraction=VALIDATION_FRACTION,
        seed=RANDOM_SEED,
    )

    if not training_samples:
        raise RuntimeError("Training split is empty.")

    positive_counts = [0] * len(LABEL_NAMES)

    for sample in training_samples:
        values = sample.labels.as_tuple

        if len(values) != len(LABEL_NAMES):
            raise ValueError("Unexpected number of labels.")

        for index, value in enumerate(values):
            if value not in (0, 1):
                raise ValueError(
                    f"Invalid value {value!r} for {LABEL_NAMES[index]}"
                )
            positive_counts[index] += int(value)

    total = len(training_samples)
    weights = {}

    print("CLASS WEIGHTS — TRAINING SPLIT ONLY")
    print("-" * 65)

    for label, positives in zip(LABEL_NAMES, positive_counts):
        negatives = total - positives

        if positives == 0:
            raise ValueError(
                f"{label} has no positive training examples; "
                "a finite positive-class weight cannot be estimated."
            )

        weight = negatives / positives
        weights[label] = weight

        print(
            f"{label:<22} "
            f"positive={positives:>2} "
            f"negative={negatives:>2} "
            f"weight={weight:.4f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "training_study_count": total,
        "validation_study_count": len(validation_samples),
        "validation_fraction": VALIDATION_FRACTION,
        "random_seed": RANDOM_SEED,
        "label_names": LABEL_NAMES,
        "positive_counts": {
            label: count
            for label, count in zip(LABEL_NAMES, positive_counts)
        },
        "positive_weights": weights,
        "positive_weights_in_label_order": [
            weights[label] for label in LABEL_NAMES
        ],
    }

    OUTPUT_PATH.write_text(
        json.dumps(result, indent=4),
        encoding="utf-8",
    )

    print(f"\nSaved: {OUTPUT_PATH}")
    print("CLASS WEIGHT CALCULATION PASSED")


if __name__ == "__main__":
    main()
