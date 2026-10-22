
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_manifest"
    / "study_series_manifest.csv"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "processed_labeled_dataset"
)

PROCESSED_CSV = PROCESSED_DIR / "processed_volumes.csv"
FAILURES_CSV = PROCESSED_DIR / "processing_failures.csv"

EXPECTED_SHAPE = (96, 160, 160)
PLANES = {"sagittal", "coronal", "axial"}

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


def validate_affine(path: Path) -> None:
    """Validate a saved 4x4 voxel-to-patient affine."""

    if not path.is_file():
        raise FileNotFoundError(f"Affine file missing: {path}")

    affine = np.load(path, allow_pickle=False)

    if affine.shape != (4, 4):
        raise ValueError(
            f"Expected affine shape (4, 4), got {affine.shape}"
        )

    if not np.isfinite(affine).all():
        raise ValueError("Affine contains NaN or infinite values.")

    if not np.allclose(
        affine[3],
        [0.0, 0.0, 0.0, 1.0],
    ):
        raise ValueError("Invalid homogeneous affine bottom row.")

    determinant = np.linalg.det(affine[:3, :3])

    if abs(determinant) <= 1e-12:
        raise ValueError("Affine is singular.")

    if not np.all(np.linalg.norm(affine[:3, :3], axis=0) > 0):
        raise ValueError("Affine has a zero-length spatial axis.")


def main() -> None:
    print("=" * 70)
    print("PROCESSED DATASET VALIDATION")
    print("=" * 70)

    for path in (MANIFEST_PATH, PROCESSED_CSV):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    manifest = pd.read_csv(MANIFEST_PATH)
    processed = pd.read_csv(PROCESSED_CSV)

    required_processed_columns = {
        "StudyInstanceUID",
        "Plane",
        "VolumePath",
        "CanonicalAffinePath",
        "ResizedAffinePath",
        *LABEL_COLUMNS,
    }

    missing_columns = required_processed_columns - set(
        processed.columns
    )

    if missing_columns:
        raise ValueError(
            f"Processed CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if "StudyInstanceUID" not in manifest.columns:
        raise ValueError(
            "Manifest has no StudyInstanceUID column."
        )

    expected_studies = set(
        manifest["StudyInstanceUID"].dropna().astype(str)
    )

    processed["StudyInstanceUID"] = (
        processed["StudyInstanceUID"].astype(str)
    )
    processed["Plane"] = (
        processed["Plane"].astype(str).str.lower().str.strip()
    )

    errors = []
    checked_volumes = 0
    checked_affines = 0

    print(f"Manifest studies: {len(expected_studies)}")
    print(f"Processed records: {len(processed)}")

    # 1. Check row count, IDs, planes, and duplicate records.
    expected_count = len(expected_studies) * len(PLANES)

    if len(processed) != expected_count:
        errors.append(
            f"Expected {expected_count} records; "
            f"found {len(processed)}."
        )

    unexpected_studies = (
        set(processed["StudyInstanceUID"]) - expected_studies
    )
    missing_studies = (
        expected_studies - set(processed["StudyInstanceUID"])
    )

    if unexpected_studies:
        errors.append(
            f"Unexpected studies: {len(unexpected_studies)}"
        )

    if missing_studies:
        errors.append(
            f"Studies with no processed records: {len(missing_studies)}"
        )

    invalid_planes = set(processed["Plane"]) - PLANES

    if invalid_planes:
        errors.append(f"Unexpected plane names: {invalid_planes}")

    duplicate_records = processed.duplicated(
        ["StudyInstanceUID", "Plane"]
    )

    if duplicate_records.any():
        errors.append(
            f"Duplicate study/plane records: "
            f"{int(duplicate_records.sum())}"
        )

    expected_pairs = {
        (study_id, plane)
        for study_id in expected_studies
        for plane in PLANES
    }

    actual_pairs = set(
        zip(
            processed["StudyInstanceUID"],
            processed["Plane"],
        )
    )

    missing_pairs = expected_pairs - actual_pairs

    print(f"Missing study/plane combinations: {len(missing_pairs)}")

    if missing_pairs:
        errors.append(
            f"Missing study/plane combinations: {len(missing_pairs)}"
        )

    # 2. Validate labels against the original manifest.
    manifest_labels = manifest.set_index(
        manifest["StudyInstanceUID"].astype(str)
    )[LABEL_COLUMNS]

    label_mismatches = 0

    for _, record in processed.iterrows():
        study_id = record["StudyInstanceUID"]

        if study_id not in manifest_labels.index:
            continue

        expected_labels = manifest_labels.loc[study_id]

        for label in LABEL_COLUMNS:
            actual = pd.to_numeric(
                pd.Series([record[label]]),
                errors="coerce",
            ).iloc[0]

            expected = pd.to_numeric(
                pd.Series([expected_labels[label]]),
                errors="coerce",
            ).iloc[0]

            if pd.isna(actual) or pd.isna(expected):
                if not (pd.isna(actual) and pd.isna(expected)):
                    label_mismatches += 1
            elif not np.isclose(actual, expected):
                label_mismatches += 1

    print(f"Label value mismatches: {label_mismatches}")

    if label_mismatches:
        errors.append(
            f"Label value mismatches: {label_mismatches}"
        )

    # 3. Validate every saved volume and both saved affine files.
    for _, record in processed.iterrows():
        study_id = record["StudyInstanceUID"]
        plane = record["Plane"]

        try:
            volume_path = Path(str(record["VolumePath"]))

            if not volume_path.is_absolute():
                volume_path = PROJECT_ROOT / volume_path

            if not volume_path.is_file():
                raise FileNotFoundError(
                    f"Volume file missing: {volume_path}"
                )

            volume = np.load(
                volume_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            if volume.shape != EXPECTED_SHAPE:
                raise ValueError(
                    f"Shape {volume.shape}; expected {EXPECTED_SHAPE}"
                )

            if volume.dtype != np.float32:
                raise ValueError(
                    f"Dtype {volume.dtype}; expected float32"
                )

            if not np.isfinite(volume).all():
                raise ValueError(
                    "Volume contains NaN or infinite values."
                )

            minimum = float(volume.min())
            maximum = float(volume.max())

            if minimum < -1e-6 or maximum > 1.000001:
                raise ValueError(
                    f"Intensity range [{minimum}, {maximum}] "
                    "is outside [0, 1]."
                )

            for column in (
                "CanonicalAffinePath",
                "ResizedAffinePath",
            ):
                affine_path = Path(str(record[column]))

                if not affine_path.is_absolute():
                    affine_path = PROJECT_ROOT / affine_path

                validate_affine(affine_path)
                checked_affines += 1

            checked_volumes += 1

        except Exception as exc:
            errors.append(
                f"{study_id} / {plane}: "
                f"{type(exc).__name__}: {exc}"
            )

    print(f"Valid volumes: {checked_volumes}/{len(processed)}")
    print(f"Valid affine files: {checked_affines}")
    print(f"Expected affine files: {len(processed) * 2}")

    # 4. Verify the processing failure report.
        
    # Read the processing failure log safely.
    if not FAILURES_CSV.exists():
        print(f"ERROR: Failure log not found: {FAILURES_CSV}")
        failures = pd.DataFrame()
    elif FAILURES_CSV.stat().st_size == 0:
        # An empty file means there are no recorded failures.
        failures = pd.DataFrame()
    else:
        try:
            failures = pd.read_csv(FAILURES_CSV)
        except pd.errors.EmptyDataError:
            failures = pd.DataFrame()

    if failures.empty:
        print("Processing failures: 0")
    else:
        print(f"Processing failures: {len(failures)}")
        print(failures.head(10).to_string(index=False))


    print("-" * 70)

    if errors:
        print(f"VALIDATION FAILED: {len(errors)} issue(s)")

        for index, error in enumerate(errors, start=1):
            print(f"{index}. {error}")

        print("=" * 70)
        sys.exit(1)

    print("All dataset validation checks passed.")
    print("PROCESSED DATASET VALIDATION PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
