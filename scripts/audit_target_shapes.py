from __future__ import annotations

import math
import traceback
from pathlib import Path

import pandas as pd

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.preprocessing.dicom.dicom_reader import DicomSeriesReader
from ish_knee.preprocessing.volume.spacing_analyzer import SpacingAnalyzer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")

OUTPUT_DIR = PROJECT_ROOT / "outputs"

AUDIT_CSV = OUTPUT_DIR / "target_shape_audit.csv"
FAILURES_CSV = OUTPUT_DIR / "target_shape_audit_failures.csv"

LABELED_AUDIT_CSV = OUTPUT_DIR / "labeled_target_shape_audit.csv"
LABELED_SUMMARY_CSV = OUTPUT_DIR / "labeled_target_shape_summary.csv"

TARGET_SPACING = (1.0, 1.0, 1.0)

TARGET_SHAPES = {
    "64x256x256": (64, 256, 256),
    "96x192x192": (96, 192, 192),
    "128x128x128": (128, 128, 128),
}

PLANES = ("sagittal", "coronal", "axial")

# Keep this list aligned with LabeledDatasetBuilder.LABELED_COLUMNS.
LABELED_COLUMNS = [
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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_volume_shape(series) -> tuple[int, int, int]:
    """Get the 3D shape using the same slice ordering as the volume builder."""
    ordered_slices = series.ordered_slices()

    if not ordered_slices:
        raise ValueError("No ordered DICOM slices found.")

    first_shape = ordered_slices[0].pixel_array.shape

    for dicom_slice in ordered_slices:
        if dicom_slice.pixel_array.shape != first_shape:
            raise ValueError(
                "Inconsistent image dimensions within series: "
                f"{dicom_slice.pixel_array.shape} != {first_shape}"
            )

    return (
        len(ordered_slices),
        int(first_shape[0]),
        int(first_shape[1]),
    )


def get_spacing_tuple(spacing) -> tuple[float, float, float]:
    """Extract spacing values in volume-axis order."""
    values = (
        float(spacing.spacing_axis_0),
        float(spacing.spacing_axis_1),
        float(spacing.spacing_axis_2),
    )

    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"Invalid voxel spacing: {values}")

    return values


def estimate_resampled_shape(
    original_shape: tuple[int, int, int],
    current_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Estimate dimensions after resampling; does not resample pixels."""
    result = []

    for size, current, target in zip(
        original_shape, current_spacing, target_spacing
    ):
        result.append(int(round(size * current / target)))

    return tuple(result)


def calculate_shape_metrics(
    resampled_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
) -> dict:
    """Calculate per-axis crop/padding and geometric voxel-grid retention."""
    crop = tuple(
        max(0, source - target)
        for source, target in zip(resampled_shape, target_shape)
    )
    padding = tuple(
        max(0, target - source)
        for source, target in zip(resampled_shape, target_shape)
    )

    source_voxels = math.prod(resampled_shape)
    target_voxels = math.prod(target_shape)

    # This estimates the fraction of the resampled voxel grid retained
    # after center cropping/padding, assuming cropping is performed per axis.
    retained_shape = tuple(
        min(source, target)
        for source, target in zip(resampled_shape, target_shape)
    )
    retained_voxels = math.prod(retained_shape)

    return {
        "crop_axis_0": crop[0],
        "crop_axis_1": crop[1],
        "crop_axis_2": crop[2],
        "padding_axis_0": padding[0],
        "padding_axis_1": padding[1],
        "padding_axis_2": padding[2],
        "resampled_voxels": source_voxels,
        "target_voxels": target_voxels,
        "target_to_resampled_voxel_ratio": (
            target_voxels / source_voxels if source_voxels > 0 else None
        ),
        "estimated_retained_voxels": retained_voxels,
        "estimated_retained_fraction": (
            retained_voxels / source_voxels if source_voxels > 0 else None
        ),
    }


def record_failure(
    failures: list[dict],
    study_uid: str,
    plane: str,
    error: Exception,
) -> None:
    """Record an error without stopping the entire audit."""
    failures.append({
        "study_uid": study_uid,
        "plane": plane,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    })


def summarize_audit(audit_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize crop/padding and retention metrics by plane and target."""
    if audit_df.empty:
        return pd.DataFrame()

    return (
        audit_df.groupby(["plane", "target_name"], dropna=False)
        .agg(
            series_count=("series_uid", "nunique"),
            mean_resampled_voxels=("resampled_voxels", "mean"),
            target_voxels=("target_voxels", "first"),
            mean_retained_fraction=("estimated_retained_fraction", "mean"),
            median_retained_fraction=("estimated_retained_fraction", "median"),
            mean_crop_axis_0=("crop_axis_0", "mean"),
            mean_crop_axis_1=("crop_axis_1", "mean"),
            mean_crop_axis_2=("crop_axis_2", "mean"),
            mean_padding_axis_0=("padding_axis_0", "mean"),
            mean_padding_axis_1=("padding_axis_1", "mean"),
            mean_padding_axis_2=("padding_axis_2", "mean"),
        )
        .reset_index()
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = DatasetPaths(
        root=DATASET_ROOT,
        train_csv=DATASET_ROOT / "train.csv",
        train_series_csv=DATASET_ROOT / "train_series.csv",
        train_series_dir=DATASET_ROOT / "train_series",
        test_csv=DATASET_ROOT / "test.csv",
        test_series_csv=DATASET_ROOT / "test_series.csv",
        test_series_dir=DATASET_ROOT / "test_series",
        sample_submission_csv=DATASET_ROOT / "sample_submission.csv",
    )

    metadata_reader = MetadataReader(paths)
    metadata_reader.validate()

    train_df = metadata_reader.train
    train_series_df = metadata_reader.train_series
    study_uids = metadata_reader.get_study_uids()

    # Confirm that the label columns expected by LabeledDatasetBuilder exist.
    missing_label_columns = [
        column for column in LABELED_COLUMNS if column not in train_df.columns
    ]
    if missing_label_columns:
        raise ValueError(
            "The training CSV is missing required label columns: "
            + ", ".join(missing_label_columns)
        )

    labeled_rows = train_df[
        train_df[LABELED_COLUMNS].notna().all(axis=1)
    ].copy()
    labeled_study_uids = set(
        labeled_rows["StudyInstanceUID"].dropna().astype(str)
    )

    print("=" * 72)
    print("RSNA KNEE TARGET SHAPE AUDIT")
    print("=" * 72)
    print(f"Training studies: {len(study_uids):,}")
    print(f"Training series: {train_series_df['SeriesInstanceUID'].nunique():,}")
    print(f"Fully labeled studies: {len(labeled_study_uids):,}")
    print(f"Target spacing: {TARGET_SPACING}")
    print(f"Candidate target shapes: {len(TARGET_SHAPES)}")
    print()

    selector = SeriesSelector(train_series_dir=paths.train_series_dir)
    dicom_reader = DicomSeriesReader()
    spacing_analyzer = SpacingAnalyzer()

    audit_rows: list[dict] = []
    failure_rows: list[dict] = []

    for index, study_uid in enumerate(study_uids, start=1):
        study_uid = str(study_uid)
        print(f"[{index}/{len(study_uids)}] Study: {study_uid}")

        try:
            selection = selector.select(study_uid, train_series_df)
        except Exception as exc:
            record_failure(failure_rows, study_uid, "study", exc)
            print(f"    Selection failed: {exc}")
            continue

        selected_series = {
            "sagittal": selection.sagittal,
            "coronal": selection.coronal,
            "axial": selection.axial,
        }

        for plane in PLANES:
            selected = selected_series[plane]

            if selected is None:
                failure_rows.append({
                    "study_uid": study_uid,
                    "plane": plane,
                    "error_type": "MissingSeries",
                    "error_message": f"No {plane} series was selected.",
                    "traceback": "",
                })
                print(f"    {plane}: no series selected")
                continue

            try:
                series = dicom_reader.read(selected.series_path)

                if series.series_instance_uid != selected.series_instance_uid:
                    raise ValueError(
                        "SeriesInstanceUID mismatch. "
                        f"Selected={selected.series_instance_uid}; "
                        f"loaded={series.series_instance_uid}"
                    )

                if series.study_instance_uid != study_uid:
                    raise ValueError(
                        "StudyInstanceUID mismatch. "
                        f"Expected={study_uid}; loaded={series.study_instance_uid}"
                    )

                original_shape = get_volume_shape(series)
                spacing_object = spacing_analyzer.analyze(series)
                current_spacing = get_spacing_tuple(spacing_object)
                resampled_shape = estimate_resampled_shape(
                    original_shape, current_spacing, TARGET_SPACING
                )

                print(
                    f"    {plane:8} original={original_shape} "
                    f"spacing={current_spacing} estimated={resampled_shape}"
                )

                for target_name, target_shape in TARGET_SHAPES.items():
                    metrics = calculate_shape_metrics(
                        resampled_shape, target_shape
                    )
                    audit_rows.append({
                        "study_uid": study_uid,
                        "series_uid": series.series_instance_uid,
                        "plane": plane,
                        "series_path": str(selected.series_path),
                        "file_count_metadata": selected.file_count,
                        "original_shape": str(original_shape),
                        "current_spacing": str(current_spacing),
                        "target_spacing": str(TARGET_SPACING),
                        "estimated_resampled_shape": str(resampled_shape),
                        "target_name": target_name,
                        "target_shape": str(target_shape),
                        **metrics,
                    })

            except Exception as exc:
                record_failure(failure_rows, study_uid, plane, exc)
                print(f"    {plane}: ERROR {type(exc).__name__}: {exc}")

    audit_columns = [
        "study_uid", "series_uid", "plane", "series_path",
        "file_count_metadata", "original_shape", "current_spacing",
        "target_spacing", "estimated_resampled_shape", "target_name",
        "target_shape", "crop_axis_0", "crop_axis_1", "crop_axis_2",
        "padding_axis_0", "padding_axis_1", "padding_axis_2",
        "resampled_voxels", "target_voxels",
        "target_to_resampled_voxel_ratio", "estimated_retained_voxels",
        "estimated_retained_fraction",
    ]
    failure_columns = [
        "study_uid", "plane", "error_type", "error_message", "traceback"
    ]

    audit_df = pd.DataFrame(audit_rows, columns=audit_columns)
    failures_df = pd.DataFrame(failure_rows, columns=failure_columns)

    audit_df.to_csv(AUDIT_CSV, index=False)
    failures_df.to_csv(FAILURES_CSV, index=False)

    # Filter the full audit using the exact complete-label rule used by
    # LabeledDatasetBuilder. This does not alter the existing train/validation split.
    labeled_audit_df = audit_df[
        audit_df["study_uid"].astype(str).isin(labeled_study_uids)
    ].copy()
    labeled_audit_df.to_csv(LABELED_AUDIT_CSV, index=False)

    labeled_summary_df = summarize_audit(labeled_audit_df)
    labeled_summary_df.to_csv(LABELED_SUMMARY_CSV, index=False)

    print()
    print("=" * 72)
    print("AUDIT SUMMARY")
    print("=" * 72)
    print(f"Studies considered: {len(study_uids):,}")
    print(f"Fully labeled studies: {len(labeled_study_uids):,}")
    print(f"Audit rows: {len(audit_df):,}")
    print(f"Failures: {len(failures_df):,}")
    print(f"Audit CSV: {AUDIT_CSV.resolve()}")
    print(f"Failures CSV: {FAILURES_CSV.resolve()}")
    print(f"Labeled audit CSV: {LABELED_AUDIT_CSV.resolve()}")
    print(f"Labeled summary CSV: {LABELED_SUMMARY_CSV.resolve()}")

    if not audit_df.empty:
        print("\nAll studies: results by target shape")
        all_summary = summarize_audit(audit_df)
        print(all_summary.to_string(index=False))

    if not labeled_audit_df.empty:
        print("\nFully labeled studies: results by plane and target shape")
        print(labeled_summary_df.to_string(index=False))
    else:
        print(
            "\nWARNING: No audit rows matched the fully labeled study UIDs. "
            "Check UID formatting and whether the selected series were audited."
        )

    if not failures_df.empty:
        print(
            "\nReview the failure CSV for missing series, invalid DICOM files, "
            "and selection errors."
        )

    print("=" * 72)


if __name__ == "__main__":
    main()
