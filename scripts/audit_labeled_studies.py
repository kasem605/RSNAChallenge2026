
from pathlib import Path

import pandas as pd

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader


DATA_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")
OUTPUT_DIR = Path("outputs/labeled_study_audit")

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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = DatasetPaths.from_root(DATA_ROOT)
    metadata = MetadataReader(paths)
    metadata.validate()

    train = metadata.train.copy()

    print("=" * 70)
    print("LABELED STUDY AUDIT")
    print("=" * 70)

    print(f"Training rows: {len(train):,}")
    print(f"Unique study IDs: {train['StudyInstanceUID'].nunique():,}")
    print()

    # Check that all expected target columns exist.
    missing_columns = [
        column for column in LABEL_COLUMNS
        if column not in train.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required label columns are missing: "
            + ", ".join(missing_columns)
        )

    # Report the number of missing values in each label.
    print("Missing values by label:")
    print(train[LABEL_COLUMNS].isna().sum().to_string())
    print()

    # A study qualifies only if all 12 labels are populated.
    complete_mask = train[LABEL_COLUMNS].notna().all(axis=1)

    complete = train.loc[complete_mask].copy()
    incomplete = train.loc[~complete_mask].copy()

    # Record exactly which labels are missing for each incomplete study.
    incomplete["MissingLabels"] = incomplete.apply(
        lambda row: ", ".join(
            column
            for column in LABEL_COLUMNS
            if pd.isna(row[column])
        ),
        axis=1,
    )

    # Save study IDs and labels for downstream dataset construction.
    complete[
        ["StudyInstanceUID", *LABEL_COLUMNS]
    ].to_csv(
        OUTPUT_DIR / "fully_labeled_studies.csv",
        index=False,
    )

    incomplete[
        ["StudyInstanceUID", "MissingLabels", *LABEL_COLUMNS]
    ].to_csv(
        OUTPUT_DIR / "incomplete_labeled_studies.csv",
        index=False,
    )

    print(f"Fully labeled studies: {len(complete):,}")
    print(f"Incomplete studies:   {len(incomplete):,}")
    print()
    print(f"Saved eligible studies to: "
          f"{OUTPUT_DIR / 'fully_labeled_studies.csv'}")
    print(f"Saved incomplete report to: "
          f"{OUTPUT_DIR / 'incomplete_labeled_studies.csv'}")

    # Validate that the two groups account for every input row.
    assert len(complete) + len(incomplete) == len(train)
    assert complete[LABEL_COLUMNS].notna().all().all()

    print()
    print("LABEL AUDIT PASSED")


if __name__ == "__main__":
    main()
