
from pathlib import Path
import csv

import numpy as np


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "outputs" / "processed_dataset_v1"
MANIFEST_PATH = DATASET_DIR / "manifest.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training_phase3d"
OUTPUT_PATH = OUTPUT_DIR / "processed_volume_quality_v1.csv"

PLANES = ["sagittal", "coronal", "axial"]


def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    with MANIFEST_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("The manifest is empty.")

    return rows


def resolve_npz_path(row):
    """
    Resolve the processed NPZ path from common manifest
    column names. Adjust this function if your manifest
    uses a different path column.
    """

    possible_columns = [
        "file_path",
        "processed_path",
        "npz_path",
        "path",
        "file",
    ]

    for column in possible_columns:
        value = row.get(column, "").strip()

        if value:
            path = Path(value)

            if not path.is_absolute():
                path = PROJECT_ROOT / path

            if path.exists():
                return path

            # Try relative to the processed dataset directory.
            alternate = DATASET_DIR / value

            if alternate.exists():
                return alternate

    # Try a filename or study UID if the manifest has one.
    for column in [
        "filename",
        "study_instance_uid",
        "study_id",
    ]:
        value = row.get(column, "").strip()

        if value:
            candidates = [
                DATASET_DIR / f"{value}.npz",
                DATASET_DIR / "train" / f"{value}.npz",
                DATASET_DIR / "validation" / f"{value}.npz",
            ]

            for path in candidates:
                if path.exists():
                    return path

    raise FileNotFoundError(
        "Could not resolve NPZ file for manifest row: "
        + repr(row)
    )


def analyze_volume(volume):
    volume = np.asarray(volume)

    if volume.size == 0:
        raise ValueError("Volume is empty.")

    if not np.issubdtype(volume.dtype, np.number):
        raise ValueError(
            f"Non-numeric volume dtype: {volume.dtype}"
        )

    finite_mask = np.isfinite(volume)
    finite_values = volume[finite_mask]

    if finite_values.size == 0:
        raise ValueError("Volume contains no finite values.")

    return {
        "shape": str(tuple(volume.shape)),
        "dtype": str(volume.dtype),
        "finite_fraction": float(np.mean(finite_mask)),
        "minimum": float(np.min(finite_values)),
        "maximum": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values)),
        "std": float(np.std(finite_values)),
        "zero_fraction": float(np.mean(volume == 0)),
        "near_zero_fraction": float(
            np.mean(np.abs(volume) < 1e-6)
        ),
        "unique_sample_count": int(
            np.unique(volume.ravel()[::max(1, volume.size // 10000)]).size
        ),
    }


def main():
    rows = load_manifest()

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}\n"
            "Rename the previous report to preserve it."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("PROCESSED MRI VOLUME QUALITY ANALYSIS")
    print("=" * 90)

    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Manifest rows: {len(rows)}")
    print()

    results = []
    errors = []

    for index, row in enumerate(rows, start=1):
        split = row.get("split", "unknown").strip().lower()

        try:
            npz_path = resolve_npz_path(row)

            with np.load(npz_path, allow_pickle=False) as data:
                for plane in PLANES:
                    if plane not in data:
                        raise KeyError(
                            f"Missing '{plane}' array in {npz_path}"
                        )

                    stats = analyze_volume(data[plane])

                    result = {
                        "study_instance_uid": (
                            row.get("study_instance_uid")
                            or row.get("study_id")
                            or row.get("StudyInstanceUID")
                            or npz_path.stem
                        ),
                        "split": split,
                        "plane": plane,
                        "file_path": str(npz_path),
                        **stats,
                        "status": "OK",
                        "error": "",
                    }

                    if stats["finite_fraction"] < 1.0:
                        result["status"] = "NON_FINITE_VALUES"
                    elif stats["std"] < 1e-6:
                        result["status"] = "NEAR_CONSTANT_VOLUME"
                    elif (
                        stats["minimum"] < -1e-5
                        or stats["maximum"] > 1.00001
                    ):
                        result["status"] = "OUTSIDE_EXPECTED_0_1_RANGE"

                    results.append(result)

        except Exception as exc:
            errors.append({
                "manifest_row": index,
                "split": split,
                "error": str(exc),
            })

        if index % 10 == 0 or index == len(rows):
            print(f"Processed manifest rows: {index}/{len(rows)}")

    fieldnames = [
        "study_instance_uid",
        "split",
        "plane",
        "file_path",
        "shape",
        "dtype",
        "finite_fraction",
        "minimum",
        "maximum",
        "mean",
        "std",
        "zero_fraction",
        "near_zero_fraction",
        "unique_sample_count",
        "status",
        "error",
    ]

    with OUTPUT_PATH.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print()
    print("-" * 90)
    print("SUMMARY")
    print("-" * 90)
    print(f"Manifest rows: {len(rows)}")
    print(f"Successful volume records: {len(results)}")
    print(f"Errors: {len(errors)}")

    for plane in PLANES:
        plane_rows = [
            result for result in results
            if result["plane"] == plane
        ]

        if not plane_rows:
            continue

        statuses = [
            result["status"] for result in plane_rows
        ]

        print()
        print(f"{plane.upper()}")
        print(f"  Volumes analyzed: {len(plane_rows)}")
        print(
            "  Shape examples: "
            + ", ".join(
                sorted(set(result["shape"] for result in plane_rows))
            )
        )
        print(
            "  Mean intensity range: "
            f"{min(result['mean'] for result in plane_rows):.4f} "
            "to "
            f"{max(result['mean'] for result in plane_rows):.4f}"
        )
        print(
            "  Standard deviation range: "
            f"{min(result['std'] for result in plane_rows):.4f} "
            "to "
            f"{max(result['std'] for result in plane_rows):.4f}"
        )
        print(
            "  Zero fraction range: "
            f"{min(result['zero_fraction'] for result in plane_rows):.1%} "
            "to "
            f"{max(result['zero_fraction'] for result in plane_rows):.1%}"
        )
        print(
            "  Non-OK statuses: "
            f"{sum(status != 'OK' for status in statuses)}"
        )

    if errors:
        print()
        print("ERRORS")
        for error in errors[:10]:
            print(
                f"  Row {error['manifest_row']}: "
                f"{error['error']}"
            )

    print()
    print(f"Report saved to: {OUTPUT_PATH}")
    print("=" * 90)


if __name__ == "__main__":
    main()
