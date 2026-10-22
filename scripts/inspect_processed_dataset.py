from pathlib import Path
import csv
import numpy as np

project_root = Path(__file__).resolve().parents[1]
manifest_path = (
    project_root / "outputs" / "processed_dataset_v1" / "manifest.csv"
)

with manifest_path.open(
    "r", newline="", encoding="utf-8-sig"
) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print("Total manifest rows:", len(rows))

split_counts = {}
for row in rows:
    split = row["split"]
    split_counts[split] = split_counts.get(split, 0) + 1

print("Split counts:", split_counts)

train_ids = {
    row["study_instance_uid"]
    for row in rows if row["split"] == "train"
}
validation_ids = {
    row["study_instance_uid"]
    for row in rows if row["split"] == "validation"
}

print("Train/validation overlap:", len(train_ids & validation_ids))

if not rows:
    raise ValueError("Manifest contains no rows.")

dataset_dir = manifest_path.parent
file_path = (dataset_dir / rows[0]["file_path"]).resolve()

if not file_path.is_relative_to(dataset_dir.resolve()):
    raise ValueError(f"File path escapes dataset directory: {file_path}")

print("First file:", file_path)

with np.load(file_path, allow_pickle=False) as data:
    print("NPZ keys:", data.files)

    for plane in ("sagittal", "coronal", "axial"):
        if plane in data.files:
            volume = data[plane]
            print(
                f"{plane}: shape={volume.shape}, "
                f"dtype={volume.dtype}, "
                f"finite={np.isfinite(volume).all()}"
            )

    if "labels" in data.files:
        labels = data["labels"]
        print("Labels:", labels.shape, labels.dtype)
    else:
        print("WARNING: NPZ does not contain a 'labels' array.")