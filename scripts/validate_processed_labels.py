
from pathlib import Path
import csv
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "processed_dataset_v1"
    / "manifest.csv"
)

PLANES = ("sagittal", "coronal", "axial")
EXPECTED_SHAPE = (128, 128, 128)

# This order must match the labels array stored in each NPZ.
LABEL_COLUMNS = (
    "label_acl",
    "label_mcl",
    "label_medial_meniscus",
    "label_lateral_meniscus",
    "label_medial_oa",
    "label_lateral_oa",
    "label_pf_oa",
    "label_effusion",
    "label_synovitis",
    "label_bakers",
    "label_contusion",
    "label_fracture",
)


def main() -> None:
    print("=" * 70)
    print("PROCESSED DATASET LABEL CONSISTENCY VALIDATION")
    print("=" * 70)

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    dataset_dir = MANIFEST_PATH.parent.resolve()

    with MANIFEST_PATH.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = set(reader.fieldnames or [])

    required_columns = {
        "study_instance_uid",
        "split",
        "file_path",
        *LABEL_COLUMNS,
    }

    missing_columns = required_columns - columns
    if missing_columns:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing_columns)}"
        )

    if not rows:
        raise ValueError("Manifest contains no studies.")

    errors = []
    seen_uids = set()
    checked_volumes = 0
    checked_labels = 0
    split_counts = {}

    for index, row in enumerate(rows, start=1):
        uid = row["study_instance_uid"]
        split = row["split"]

        if not uid:
            errors.append(f"Row {index}: empty study UID")
            continue

        if uid in seen_uids:
            errors.append(f"Duplicate study UID: {uid}")
        seen_uids.add(uid)

        split_counts[split] = split_counts.get(split, 0) + 1

        relative_path = Path(row["file_path"])
        file_path = (dataset_dir / relative_path).resolve()

        if not file_path.is_relative_to(dataset_dir):
            errors.append(
                f"{uid}: file path escapes dataset directory"
            )
            continue

        if not file_path.is_file():
            errors.append(f"{uid}: missing file {file_path}")
            continue

        try:
            manifest_labels = np.asarray(
                [float(row[name]) for name in LABEL_COLUMNS],
                dtype=np.float32,
            )
        except (TypeError, ValueError) as exc:
            errors.append(
                f"{uid}: invalid manifest label value: {exc}"
            )
            continue

        if not np.isin(manifest_labels, (0.0, 1.0)).all():
            errors.append(
                f"{uid}: manifest labels are not all binary"
            )
            continue

        try:
            with np.load(file_path, allow_pickle=False) as data:
                missing_keys = (
                    set(PLANES) | {"labels"}
                ) - set(data.files)

                if missing_keys:
                    errors.append(
                        f"{uid}: NPZ missing keys "
                        f"{sorted(missing_keys)}"
                    )
                    continue

                labels = data["labels"].copy()

                for plane in PLANES:
                    volume = data[plane]

                    if volume.shape != EXPECTED_SHAPE:
                        errors.append(
                            f"{uid}: {plane} shape is "
                            f"{volume.shape}, expected {EXPECTED_SHAPE}"
                        )
                        continue

                    if volume.dtype != np.float32:
                        errors.append(
                            f"{uid}: {plane} dtype is {volume.dtype}, "
                            "expected float32"
                        )
                        continue

                    if not np.isfinite(volume).all():
                        errors.append(
                            f"{uid}: {plane} contains non-finite values"
                        )
                        continue

                    checked_volumes += 1

                if labels.shape != (len(LABEL_COLUMNS),):
                    errors.append(
                        f"{uid}: labels shape is {labels.shape}, "
                        f"expected ({len(LABEL_COLUMNS)},)"
                    )
                    continue

                if not np.isfinite(labels).all():
                    errors.append(f"{uid}: labels contain non-finite values")
                    continue

                if not np.isin(labels, (0.0, 1.0)).all():
                    errors.append(f"{uid}: NPZ labels are not binary")
                    continue

                if not np.array_equal(labels, manifest_labels):
                    differences = [
                        LABEL_COLUMNS[i]
                        for i in range(len(LABEL_COLUMNS))
                        if labels[i] != manifest_labels[i]
                    ]
                    errors.append(
                        f"{uid}: labels disagree with manifest: "
                        f"{differences}"
                    )
                    continue

                checked_labels += 1

        except Exception as exc:
            errors.append(f"{uid}: failed to read NPZ: {exc}")

        if index % 10 == 0 or index == len(rows):
            print(f"Checked manifest rows: {index}/{len(rows)}")

    print("\nSummary")
    print("-" * 70)
    print(f"Manifest studies:       {len(rows)}")
    print(f"Split counts:           {split_counts}")
    print(f"Unique study IDs:       {len(seen_uids)}")
    print(f"Volumes validated:      {checked_volumes}")
    print(f"Label arrays matched:   {checked_labels}/{len(rows)}")
    print(f"Errors:                 {len(errors)}")

    if errors:
        print("\nValidation errors:")
        for error in errors:
            print(f" - {error}")

        raise SystemExit(
            "\nVALIDATION FAILED. Do not start training yet."
        )

    print("\nPROCESSED DATASET VALIDATION PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
