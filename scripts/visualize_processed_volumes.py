
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "outputs" / "processed_dataset_v1"
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training_phase3d"
OUTPUT_IMAGE = OUTPUT_DIR / "processed_volume_visualization.png"

PLANES = ["sagittal", "coronal", "axial"]


def resolve_npz_path(row: pd.Series) -> Path:
    """Find the NPZ file associated with a manifest row."""

    # Try common path-column names first.
    for column in [
        "file_path",
        "processed_path",
        "npz_path",
        "path",
        "file",
    ]:
        if column in row.index and pd.notna(row[column]):
            candidate = Path(str(row[column]))

            if candidate.is_absolute() and candidate.exists():
                return candidate

            candidate = PROJECT_ROOT / candidate
            if candidate.exists():
                return candidate

            candidate = PROCESSED_DIR / str(row[column])
            if candidate.exists():
                return candidate

    # Otherwise, construct the path from split and study identifier.
    split = str(row.get("split", "")).strip().lower()

    study_id = None
    for column in [
        "study_instance_uid",
        "study_uid",
        "study_id",
        "StudyInstanceUID",
    ]:
        if column in row.index and pd.notna(row[column]):
            study_id = str(row[column])
            break

    if not study_id:
        raise ValueError(
            f"Cannot identify study in manifest row: {row.to_dict()}"
        )

    # Check both common filename conventions.
    for folder in [PROCESSED_DIR / split, PROCESSED_DIR]:
        for filename in [
            f"{study_id}.npz",
            f"{study_id.replace('.', '_')}.npz",
        ]:
            candidate = folder / filename
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"Cannot find NPZ file for study {study_id}, split={split}. "
        f"Manifest columns: {list(row.index)}"
    )


def load_volumes(npz_path: Path) -> dict[str, np.ndarray]:
    """Load the three processed imaging volumes."""

    with np.load(npz_path, allow_pickle=False) as data:
        missing = [plane for plane in PLANES if plane not in data.files]

        if missing:
            raise KeyError(
                f"{npz_path.name} is missing arrays: {missing}. "
                f"Available arrays: {data.files}"
            )

        volumes = {
            plane: np.asarray(data[plane], dtype=np.float32)
            for plane in PLANES
        }

    for plane, volume in volumes.items():
        if volume.ndim != 3:
            raise ValueError(
                f"{plane} must be 3D, got shape {volume.shape}"
            )

    return volumes


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    if OUTPUT_IMAGE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_IMAGE}\n"
            "Rename the existing image or change OUTPUT_IMAGE "
            "so this script does not overwrite it."
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    if "split" not in manifest.columns:
        raise KeyError(
            f"Manifest needs a 'split' column. "
            f"Found: {list(manifest.columns)}"
        )

    # Select up to two studies from each split.
    selected_rows = []

    for split in ["train", "validation"]:
        subset = manifest[
            manifest["split"].astype(str).str.lower() == split
        ]

        if subset.empty:
            print(f"WARNING: No studies found for split '{split}'.")
            continue

        selected_rows.extend(
            [row for _, row in subset.head(2).iterrows()]
        )

    if not selected_rows:
        raise RuntimeError("No studies selected for visualization.")

    study_images = []

    for row in selected_rows:
        npz_path = resolve_npz_path(row)
        volumes = load_volumes(npz_path)

        study_id = str(
            row.get(
                "study_instance_uid",
                row.get("study_uid", row.get("study_id", npz_path.stem)),
            )
        )

        split = str(row["split"]).lower()

        print(f"\nStudy: {study_id}")
        print(f"Split: {split}")
        print(f"File:  {npz_path}")

        for plane, volume in volumes.items():
            print(
                f"  {plane:8s} shape={volume.shape}, "
                f"min={volume.min():.4f}, "
                f"max={volume.max():.4f}, "
                f"mean={volume.mean():.4f}"
            )

        study_images.append((split, study_id, volumes))

    # -----------------------------------------------------
    # Plot the central slice along axis 0 for each volume.
    # This is an initial visual quality check, not a claim
    # about the anatomical orientation of that axis.
    # -----------------------------------------------------

    rows = len(study_images)
    fig, axes = plt.subplots(
        rows,
        3,
        figsize=(12, 4 * rows),
        squeeze=False,
    )

    for row_index, (split, study_id, volumes) in enumerate(study_images):
        for column_index, plane in enumerate(PLANES):
            ax = axes[row_index, column_index]
            volume = volumes[plane]

            middle_index = volume.shape[0] // 2
            image_slice = volume[middle_index, :, :]

            ax.imshow(
                image_slice,
                cmap="gray",
                origin="lower",
                vmin=0.0,
                vmax=1.0,
            )

            ax.set_title(
                f"{split.upper()} | {plane.capitalize()}\n"
                f"Slice {middle_index} — {study_id[:18]}"
            )
            ax.set_xlabel("Array column")
            ax.set_ylabel("Array row")
            ax.set_aspect("equal")

    fig.suptitle(
        "Processed Knee MRI Volumes — Central Slice Review",
        fontsize=14,
    )
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches="tight")

    print(f"\nVisualization saved to:\n{OUTPUT_IMAGE}")
    print("\nReview the images for visible anatomy, excessive padding,")
    print("unexpected blank images, and obvious orientation problems.")

    plt.show()


if __name__ == "__main__":
    main()
