# `scripts/validate_dataset_manifest.py`

from pathlib import Path

import pandas as pd

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MANIFEST_PATH = (
    OUTPUT_DIR
    / "dataset_manifest"
    / "study_series_manifest.csv"
)

LABELED_STUDIES_PATH = (
    OUTPUT_DIR
    / "labeled_study_audit"
    / "fully_labeled_studies.csv"
)

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
    print("=" * 70)
    print("DATASET MANIFEST VALIDATION")
    print("=" * 70)

    for path in [MANIFEST_PATH, LABELED_STUDIES_PATH]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    manifest = pd.read_csv(MANIFEST_PATH)
    labeled = pd.read_csv(LABELED_STUDIES_PATH)

    print(f"Manifest rows: {len(manifest)}")
    print(f"Fully labeled source rows: {len(labeled)}")

    # Verify required manifest columns.
    required_columns = [
        "AllPlanesAvailable",
        "MissingPlanes",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in manifest.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Manifest is missing columns: {missing_columns}"
        )

    # Identify the study ID column.
    possible_id_columns = [
        "study_id",
        "StudyInstanceUID",
        "study_instance_uid",
        "StudyInstanceUID_x",
    ]

    study_id_column = next(
        (
            col for col in possible_id_columns
            if col in manifest.columns
        ),
        None,
    )

    if study_id_column is None:
        raise ValueError(
            "Could not identify study ID column. "
            f"Available columns: {list(manifest.columns)}"
        )

    source_id_column = next(
        (
            col for col in possible_id_columns
            if col in labeled.columns
        ),
        None,
    )

    if source_id_column is None:
        raise ValueError(
            "Could not identify study ID column "
            "in fully_labeled_studies.csv."
        )

    manifest_ids = set(
        manifest[study_id_column].dropna().astype(str)
    )
    source_ids = set(
        labeled[source_id_column].dropna().astype(str)
    )

    duplicate_count = int(
        manifest[study_id_column].duplicated().sum()
    )

    missing_studies = source_ids - manifest_ids
    unexpected_studies = manifest_ids - source_ids

    print(f"Unique manifest studies: {len(manifest_ids)}")
    print(f"Duplicate manifest rows: {duplicate_count}")
    print(f"Missing studies: {len(missing_studies)}")
    print(f"Unexpected studies: {len(unexpected_studies)}")

    # Verify all three selections are reported as available.
    all_planes = (
        manifest["AllPlanesAvailable"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    print(f"Studies with all planes: {int(all_planes.sum())}")
    print(f"Studies missing a plane: {int((~all_planes).sum())}")

    # Check that each expected series selection is present.
    series_columns = {
        "sagittal": [
            "sagittal_SeriesInstanceUID",
            "sagittal_Path",
        ],
        "coronal": [
            "coronal_SeriesInstanceUID",
            "coronal_Path",
        ],
        "axial": [
            "axial_SeriesInstanceUID",
            "axial_Path",
        ],
    }

    selected_columns = {}

    for plane, candidates in series_columns.items():
        column = next(
            (c for c in candidates if c in manifest.columns),
            None,
        )

        if column is None:
            print(
                f"WARNING: Could not identify the {plane} "
                f"series UID/path column. Review manifest columns."
            )
        else:
            selected_columns[plane] = column
            populated = manifest[column].notna() & (
                manifest[column].astype(str).str.strip() != ""
            )
            print(
                f"{plane.title()} selection populated: "
                f"{int(populated.sum())}/{len(manifest)} "
                f"(column: {column})"
            )

    # Compare label values where labels are included in the manifest.
    comparable_labels = [
        col for col in LABEL_COLUMNS
        if col in manifest.columns and col in labeled.columns
    ]

    if comparable_labels:
        source_labels = labeled.set_index(
            labeled[source_id_column].astype(str)
        )

        manifest_labels = manifest.set_index(
            manifest[study_id_column].astype(str)
        )

        mismatch_count = 0

        for study_id in manifest_ids & source_ids:
            for column in comparable_labels:
                actual = manifest_labels.at[study_id, column]
                expected = source_labels.at[study_id, column]

                if isinstance(actual, pd.Series):
                    actual = actual.iloc[0]
                if isinstance(expected, pd.Series):
                    expected = expected.iloc[0]

                if pd.isna(actual) and pd.isna(expected):
                    continue

                if pd.isna(actual) != pd.isna(expected):
                    mismatch_count += 1
                elif str(actual).strip() != str(expected).strip():
                    mismatch_count += 1

        print(f"Label value mismatches: {mismatch_count}")
    else:
        mismatch_count = None
        print(
            "Label comparison skipped: no matching label "
            "columns were found in both files."
        )

    # Final validation.
    passed = (
        len(manifest) == len(source_ids)
        and len(manifest_ids) == len(source_ids)
        and duplicate_count == 0
        and not missing_studies
        and not unexpected_studies
        and all_planes.all()
        and mismatch_count in (0, None)
    )

    print("-" * 70)

    if passed:
        print("MANIFEST VALIDATION PASSED")
    else:
        print("MANIFEST VALIDATION NEEDS REVIEW")

    print("=" * 70)

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
