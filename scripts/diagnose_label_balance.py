
from pathlib import Path

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder
from ish_knee.dataset.labeled_dataset_builder import LabeledDatasetBuilder
from ish_knee.model.study_split import StudySplit


DATA_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")

LABEL_COLUMNS = [
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


def main():
    paths = DatasetPaths.from_root(DATA_ROOT)
    metadata = MetadataReader(paths)
    metadata.validate()

    selector = SeriesSelector(paths.train_series_dir)
    sample_builder = DatasetSampleBuilder(metadata, selector)
    labeled_builder = LabeledDatasetBuilder(sample_builder)

    samples = labeled_builder.build_all()

    training_samples, validation_samples = StudySplit.split(
        samples,
        validation_fraction=0.30,
        seed=42,
    )

    print(f"Training studies:   {len(training_samples)}")
    print(f"Validation studies: {len(validation_samples)}")
    print()
    print(
        f"{'Label':<22}"
        f"{'Positive':>10}"
        f"{'Negative':>10}"
        f"{'Pos. rate':>12}"
    )
    print("-" * 54)

    for column in LABEL_COLUMNS:
        positives = sum(
            int(getattr(sample.labels, {
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
            }[column]))
            for sample in training_samples
        )

        total = len(training_samples)
        negatives = total - positives
        rate = positives / total if total else 0.0

        print(
            f"{column:<22}"
            f"{positives:>10}"
            f"{negatives:>10}"
            f"{rate:>11.1%}"
        )


if __name__ == "__main__":
    main()
