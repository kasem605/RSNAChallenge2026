from pathlib import Path

import numpy as np
import pandas as pd
import pydicom


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")
SERIES_DIR = DATASET_ROOT / "train_series"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "dicom_series_geometry_audit.csv"
)

STUDY_UID = (
    "1.2.826.0.1.3680043.8.498."
    "10004873229099053869093324292195817260"
)

DIRECTION_TOLERANCE = 1e-3
SPACING_TOLERANCE_MM = 0.5


def inspect_series(series_path: Path) -> dict:
    records = []

    for path in series_path.iterdir():
        if not path.is_file():
            continue

        try:
            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
            )

            if not hasattr(ds, "ImageOrientationPatient"):
                continue

            if not hasattr(ds, "ImagePositionPatient"):
                continue

            orientation = np.asarray(
                ds.ImageOrientationPatient,
                dtype=float,
            )

            position = np.asarray(
                ds.ImagePositionPatient,
                dtype=float,
            )

            if orientation.shape != (6,) or position.shape != (3,):
                continue

            if not np.isfinite(orientation).all():
                continue

            if not np.isfinite(position).all():
                continue

            row_direction = orientation[:3]
            column_direction = orientation[3:]
            normal = np.cross(row_direction, column_direction)

            records.append({
                "path": str(path),
                "instance": getattr(ds, "InstanceNumber", None),
                "row": row_direction,
                "column": column_direction,
                "normal": normal,
                "position": position,
            })

        except Exception:
            continue

    result = {
        "series_uid": series_path.name,
        "files_with_geometry": len(records),
    }

    if len(records) < 2:
        result["status"] = "INSUFFICIENT_GEOMETRY"
        return result

    reference = records[0]

    row_deviations = [
        np.linalg.norm(item["row"] - reference["row"])
        for item in records
    ]

    column_deviations = [
        np.linalg.norm(item["column"] - reference["column"])
        for item in records
    ]

    normal = reference["normal"]
    normal_length = np.linalg.norm(normal)

    if normal_length < 1e-8:
        result["status"] = "INVALID_REFERENCE_ORIENTATION"
        return result

    normal = normal / normal_length

    # Sort by physical position along the slice normal,
    # rather than relying on InstanceNumber.
    projected_positions = np.asarray([
        np.dot(item["position"], normal)
        for item in records
    ])

    order = np.argsort(projected_positions)
    sorted_positions = projected_positions[order]
    gaps = np.diff(sorted_positions)

    duplicate_positions = int(
        np.sum(np.abs(gaps) < 1e-4)
    )

    positive_gaps = gaps[gaps > 1e-4]

    median_spacing = (
        float(np.median(positive_gaps))
        if len(positive_gaps)
        else np.nan
    )

    max_spacing_deviation = (
        float(np.max(np.abs(positive_gaps - median_spacing)))
        if len(positive_gaps) and np.isfinite(median_spacing)
        else np.nan
    )

    result.update({
        "row_direction_max_deviation": float(
            max(row_deviations)
        ),
        "column_direction_max_deviation": float(
            max(column_deviations)
        ),
        "orientation_consistent": (
            max(row_deviations) <= DIRECTION_TOLERANCE
            and max(column_deviations) <= DIRECTION_TOLERANCE
        ),
        "duplicate_slice_positions": duplicate_positions,
        "median_slice_spacing_mm": median_spacing,
        "max_spacing_deviation_mm": max_spacing_deviation,
        "spacing_regular": (
            bool(max_spacing_deviation <= SPACING_TOLERANCE_MM)
            if np.isfinite(max_spacing_deviation)
            else False
        ),
        "status": "OK",
    })

    return result


def main():
    study_path = SERIES_DIR / STUDY_UID

    if not study_path.exists():
        raise FileNotFoundError(study_path)

    results = []

    for series_path in sorted(study_path.iterdir()):
        if series_path.is_dir():
            results.append(inspect_series(series_path))

    report = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(OUTPUT_PATH, index=False)

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 220,
    ):
        print(report.to_string(index=False))

    print(f"\nReport saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()