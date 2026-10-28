
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "processed_dataset_v1"
    / "manifest.csv"
)


def main():
    with MANIFEST_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    print("Manifest columns:", list(rows[0].keys()))

    train_rows = [
        row for row in rows
        if row["split"] == "train"
    ]

    if not train_rows:
        raise ValueError("No training studies found in manifest.")

    print(f"Training studies: {len(train_rows)}")

    # Inspect the first training study.
    row = train_rows[0]

    npz_path = Path(row["file_path"])

    # Support manifest paths stored relative to the dataset directory.
    if not npz_path.is_absolute():
        npz_path = MANIFEST_PATH.parent / npz_path

    if not npz_path.is_file():
        raise FileNotFoundError(
            f"Processed volume not found: {npz_path}"
        )

    #print(f"Study UID: {row.get('study_uid', 'Unknown')}")
    print(f"Study UID: {row['study_instance_uid']}")
    print(f"NPZ file: {npz_path}")

    with np.load(npz_path) as data:
        for plane in ("sagittal", "coronal", "axial"):
            if plane not in data:
                raise KeyError(
                    f"Missing '{plane}' in {npz_path.name}"
                )

            volume = data[plane].astype(np.float32)

            print(f"\n{plane.capitalize()}")
            print(f"Shape: {volume.shape}")
            print(f"Min: {volume.min():.4f}")
            print(f"Max: {volume.max():.4f}")
            print(f"Mean: {volume.mean():.4f}")
            print(f"Std: {volume.std():.4f}")

            # Display three central slices from the 3D array.
            mid = volume.shape[0] // 2
            indices = [
                max(0, mid - 16),
                mid,
                min(volume.shape[0] - 1, mid + 16),
            ]

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            for ax, index in zip(axes, indices):
                ax.imshow(
                    volume[index],
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
                ax.set_title(f"Slice {index}")
                ax.axis("off")

            fig.suptitle(f"{plane.capitalize()} volume")
            plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
