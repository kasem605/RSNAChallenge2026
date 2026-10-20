
from pathlib import Path
import csv

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "outputs" / "processed_dataset_v1"

EXPECTED_SHAPE = (128, 128, 128)
EXPECTED_LABEL_COUNT = 12
EXPECTED_SPLITS = {"train": 41, "validation": 17}


def main() -> None:
    manifest_path = DATASET_DIR / "manifest.csv"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    if len(rows) != sum(EXPECTED_SPLITS.values()):
        raise ValueError(f"Expected 58 manifest rows; found {len(rows)}")

    uids = [row["study_instance_uid"] for row in rows]
    if len(uids) != len(set(uids)):
        raise ValueError("Duplicate study UIDs found in manifest")

    for split, expected_count in EXPECTED_SPLITS.items():
        actual_count = sum(row["split"] == split for row in rows)
        if actual_count != expected_count:
            raise ValueError(
                f"{split}: expected {expected_count}, found {actual_count}"
            )

    train_uids = {
        row["study_instance_uid"] for row in rows if row["split"] == "train"
    }
    validation_uids = {
        row["study_instance_uid"]
        for row in rows
        if row["split"] == "validation"
    }

    if train_uids & validation_uids:
        raise ValueError("Training/validation leakage detected")

    checked = 0

    for row in rows:
        file_path = DATASET_DIR / row["file_path"]

        if not file_path.is_file():
            raise FileNotFoundError(f"Missing study file: {file_path}")

        with np.load(file_path, allow_pickle=False) as data:
            expected_keys = {"sagittal", "coronal", "axial", "labels"}

            if set(data.files) != expected_keys:
                raise ValueError(
                    f"{file_path.name}: unexpected keys {data.files}"
                )

            for plane in ("sagittal", "coronal", "axial"):
                volume = data[plane]

                if volume.shape != EXPECTED_SHAPE:
                    raise ValueError(
                        f"{file_path.name} {plane}: shape {volume.shape}"
                    )

                if volume.dtype != np.float32:
                    raise TypeError(
                        f"{file_path.name} {plane}: dtype {volume.dtype}"
                    )

                if not np.isfinite(volume).all():
                    raise ValueError(
                        f"{file_path.name} {plane}: non-finite values"
                    )

            labels = data["labels"]

            if labels.shape != (EXPECTED_LABEL_COUNT,):
                raise ValueError(
                    f"{file_path.name}: invalid label shape {labels.shape}"
                )

            if not np.isin(labels, (0, 1)).all():
                raise ValueError(
                    f"{file_path.name}: labels are not binary"
                )

        checked += 1
        print(f"PASS {checked:02d}/58 [{row['split']}] {row['study_instance_uid']}")

    print("\n" + "=" * 60)
    print("PROCESSED DATASET VALIDATION PASSED")
    print(f"Training studies:   {len(train_uids)}")
    print(f"Validation studies: {len(validation_uids)}")
    print(f"Verified files:     {checked}")
    print("=" * 60)


if __name__ == "__main__":
    main()
