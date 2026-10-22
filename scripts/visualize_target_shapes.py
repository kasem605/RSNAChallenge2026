from __future__ import annotations

"""Visual review of target shapes for labeled RSNA knee MRI studies.

This script saves montage PNGs under outputs/target_shape_review.
It does not modify outputs/processed_dataset_v1.

Orientation note: this first review uses the stored volume orientation. The
OrientationTransformer is not applied because its OrientationTransform must be
provided by the project's OrientationAnalyzer; connect that exact API before
interpreting images as standardized anatomical views.
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.preprocessing.dicom.dicom_reader import DicomSeriesReader
from ish_knee.preprocessing.volume.spacing_analyzer import SpacingAnalyzer
from ish_knee.preprocessing.dicom.dicom_volume_builder import DicomVolumeBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "target_shape_review"
TARGET_SPACING = (1.0, 1.0, 1.0)
TARGET_SHAPES = {
    "64x256x256": (64, 256, 256),
    "96x192x192": (96, 192, 192),
    "128x128x128": (128, 128, 128),
}
PLANES = ("sagittal", "coronal", "axial")
LABELED_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
    "Baker's", "Contusion", "Fracture",
]


def get_spacing_tuple(spacing) -> tuple[float, float, float]:
    values = (
        float(spacing.spacing_axis_0),
        float(spacing.spacing_axis_1),
        float(spacing.spacing_axis_2),
    )
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError(f"Invalid voxel spacing: {values}")
    return values


def resample_volume(volume, current_spacing, target_spacing):
    """Linearly resample the volume to the target voxel spacing."""
    factors = tuple(c / t for c, t in zip(current_spacing, target_spacing))
    return np.asarray(
        zoom(volume, zoom=factors, order=1, mode="nearest"),
        dtype=np.float32,
    )


def center_crop_or_pad(volume: np.ndarray, target_shape: tuple[int, int, int]):
    """Center crop and/or zero-pad without changing the input volume."""
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume.shape}")

    slices = []
    for source_size, target_size in zip(volume.shape, target_shape):
        if source_size > target_size:
            start = (source_size - target_size) // 2
            slices.append(slice(start, start + target_size))
        else:
            slices.append(slice(None))
    cropped = volume[tuple(slices)]

    pad_width = []
    for current_size, target_size in zip(cropped.shape, target_shape):
        deficit = max(0, target_size - current_size)
        before = deficit // 2
        pad_width.append((before, deficit - before))

    return np.ascontiguousarray(
        np.pad(cropped, pad_width, mode="constant", constant_values=0),
        dtype=np.float32,
    )


def display_limits(volume):
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, (1, 99))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def save_montage(resampled, target_volumes, study_uid, plane, series_uid, output_path):
    volumes = [("Resampled 1 mm", resampled), *target_volumes.items()]
    fig, axes = plt.subplots(3, len(volumes), figsize=(4 * len(volumes), 10), squeeze=False)
    fig.suptitle(
        f"{plane.upper()} | Study {study_uid}\nSeries {series_uid} "
        "(stored array orientation; not standardized)",
        fontsize=11,
    )

    for col, (label, vol) in enumerate(volumes):
        vmin, vmax = display_limits(vol)
        for axis in range(3):
            index = vol.shape[axis] // 2
            image = (
                vol[index, :, :] if axis == 0 else
                vol[:, index, :] if axis == 1 else
                vol[:, :, index]
            )
            ax = axes[axis, col]
            ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"{label}\nshape={vol.shape}\naxis {axis}, slice {index}")
            ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
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

    metadata = MetadataReader(paths)
    metadata.validate()
    train_df = metadata.train
    train_series_df = metadata.train_series

    missing = [c for c in LABELED_COLUMNS if c not in train_df.columns]
    if missing:
        raise ValueError(f"Training CSV is missing label columns: {missing}")

    labeled = train_df[train_df[LABELED_COLUMNS].notna().all(axis=1)].copy()
    labeled["StudyInstanceUID"] = labeled["StudyInstanceUID"].astype(str)
    if labeled.empty:
        raise ValueError("No fully labeled studies found.")

    # Review up to three fully labeled studies. Change to explicit UIDs if desired.
    study_uids = labeled["StudyInstanceUID"].head(3).tolist()
    selector = SeriesSelector(train_series_dir=paths.train_series_dir)
    reader = DicomSeriesReader()
    volume_builder = DicomVolumeBuilder()
    spacing_analyzer = SpacingAnalyzer()
    manifest = []

    print(f"Fully labeled studies: {len(labeled)}")
    print(f"Studies selected for visual review: {len(study_uids)}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    for study_index, study_uid in enumerate(study_uids, start=1):
        print(f"\n[{study_index}/{len(study_uids)}] Study {study_uid}")
        selection = selector.select(study_uid, train_series_df)
        selected_by_plane = {
            "sagittal": selection.sagittal,
            "coronal": selection.coronal,
            "axial": selection.axial,
        }

        for plane in PLANES:
            selected = selected_by_plane[plane]
            if selected is None:
                print(f"  {plane}: no selected series")
                manifest.append({
                    "study_uid": study_uid, "plane": plane,
                    "status": "skipped", "reason": "No series selected",
                })
                continue

            try:
                series = reader.read(selected.series_path)
                if str(series.study_instance_uid) != study_uid:
                    raise ValueError(
                        f"Study UID mismatch: expected {study_uid}, "
                        f"loaded {series.study_instance_uid}"
                    )
                if str(series.series_instance_uid) != str(selected.series_instance_uid):
                    raise ValueError(
                        f"Series UID mismatch: expected {selected.series_instance_uid}, "
                        f"loaded {series.series_instance_uid}"
                    )

                dicom_volume = volume_builder.build(series)
                source_volume = np.asarray(dicom_volume.volume)
                spacing = get_spacing_tuple(spacing_analyzer.analyze(series))
                resampled = resample_volume(source_volume, spacing, TARGET_SPACING)

                target_volumes = {
                    name: center_crop_or_pad(resampled, shape)
                    for name, shape in TARGET_SHAPES.items()
                }
                image_path = (
                    OUTPUT_DIR / study_uid /
                    f"{plane}_{series.series_instance_uid}.png"
                )
                save_montage(
                    resampled, target_volumes, study_uid, plane,
                    str(series.series_instance_uid), image_path,
                )
                manifest.append({
                    "study_uid": study_uid,
                    "series_uid": str(series.series_instance_uid),
                    "plane": plane,
                    "source_shape": str(tuple(source_volume.shape)),
                    "source_spacing": str(spacing),
                    "resampled_shape": str(tuple(resampled.shape)),
                    "output_image": str(image_path),
                    "status": "success",
                    "orientation_standardized": False,
                })
                print(f"  {plane}: saved {image_path}")
                del source_volume, resampled, target_volumes, dicom_volume

            except Exception as exc:
                manifest.append({
                    "study_uid": study_uid,
                    "series_uid": str(selected.series_instance_uid),
                    "plane": plane,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                print(f"  {plane}: ERROR {type(exc).__name__}: {exc}")

    manifest_path = OUTPUT_DIR / "visualization_manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    print("\n" + "=" * 72)
    print("TARGET SHAPE VISUALIZATION COMPLETE")
    print("=" * 72)
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")
    print("Note: stored axes are not standardized anatomical directions.")
    print("This script does not modify outputs/processed_dataset_v1.")


if __name__ == "__main__":
    main()
