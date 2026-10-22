from pathlib import Path

import numpy as np
import pandas as pd
import pydicom


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")

SERIES_CSV = DATASET_ROOT / "train_series.csv"
SERIES_DIR = DATASET_ROOT / "train_series"


def inspect_series(series_path: Path) -> dict:
    files = list(series_path.glob("*"))

    dicom_files = []
    for path in files:
        if not path.is_file():
            continue

        try:
            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=False,
            )
            if hasattr(ds, "ImageOrientationPatient"):
                dicom_files.append(ds)
        except Exception:
            continue

    if not dicom_files:
        return {
            "series_uid": series_path.name,
            "status": "NO_ORIENTATION_METADATA",
        }

    ds = dicom_files[0]

    orientation = np.asarray(
        ds.ImageOrientationPatient,
        dtype=float,
    )

    row_direction = orientation[:3]
    column_direction = orientation[3:]
    slice_normal = np.cross(row_direction, column_direction)

    def direction_label(vector):
        axis_names = ("X", "Y", "Z")
        index = int(np.argmax(np.abs(vector)))
        sign = "+" if vector[index] >= 0 else "-"
        return f"{sign}{axis_names[index]}"

    return {
        "series_uid": series_path.name,
        "dicom_files_with_orientation": len(dicom_files),
        "row_direction": tuple(np.round(row_direction, 4)),
        "column_direction": tuple(np.round(column_direction, 4)),
        "slice_normal": tuple(np.round(slice_normal, 4)),
        "row_dominant_axis": direction_label(row_direction),
        "column_dominant_axis": direction_label(column_direction),
        "slice_dominant_axis": direction_label(slice_normal),
        "status": "OK",
    }


def main():
    metadata = pd.read_csv(SERIES_CSV)

    print("=" * 70)
    print("DICOM ORIENTATION AUDIT")
    print("=" * 70)
    print(f"Series metadata rows: {len(metadata):,}")

    # Start with the study used in your previous orientation test.
    study_uid = (
        "1.2.826.0.1.3680043.8.498."
        "10004873229099053869093324292195817260"
    )

    study_dir = SERIES_DIR / study_uid

    if not study_dir.exists():
        raise FileNotFoundError(
            f"Study directory not found: {study_dir}"
        )

    results = []

    for series_path in sorted(study_dir.iterdir()):
        if series_path.is_dir():
            result = inspect_series(series_path)
            results.append(result)

            print("\n" + "-" * 70)
            for key, value in result.items():
                print(f"{key}: {value}")

    if results:
        output_path = (
            PROJECT_ROOT
            / "outputs"
            / "orientation_audit_sample.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f"\nAudit report: {output_path}")

    print("\nAUDIT COMPLETE")


if __name__ == "__main__":
    main()