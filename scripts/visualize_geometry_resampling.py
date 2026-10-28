
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.preprocessing.dicom.dicom_reader import (
    DicomSeriesReader,
)
from ish_knee.preprocessing.dicom.dicom_volume_builder import (
    DicomVolumeBuilder,
)
from ish_knee.preprocessing.geometry.voxel_to_patient_affine import (
    calculate_voxel_to_patient_affine,
)
from ish_knee.preprocessing.geometry.canonical_grid import (
    build_canonical_grid,
)
from ish_knee.preprocessing.geometry.volume_resampler import (
    VolumeResampler,
)


DATASET_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")
OUTPUT_ROOT = Path("outputs/geometry_resampling_review")

# Target array axes are [LPS X, LPS Y, LPS Z].
# Each plane is displayed at the corresponding central coordinate.
PLANE_TARGET_AXIS = {
    "sagittal": 0,  # Fixed LPS X; display Y-Z
    "coronal": 1,   # Fixed LPS Y; display X-Z
    "axial": 2,     # Fixed LPS Z; display X-Y
}


def window_image(image: np.ndarray) -> np.ndarray:
    """Robust display scaling without changing the saved voxel data."""
    image = np.asarray(image, dtype=np.float32)
    valid = image[np.isfinite(image)]

    if valid.size == 0:
        return np.zeros_like(image)

    low, high = np.percentile(valid, [1, 99])

    if high <= low:
        return np.zeros_like(image)

    return np.clip((image - low) / (high - low), 0, 1)


def extract_source_slice(
    volume: np.ndarray,
    plane: str,
    slice_index: int,
) -> np.ndarray:
    """Extract a source slice in native [slice, row, column] order."""
    if plane in ("sagittal", "coronal", "axial"):
        # The selected source series is already oriented to this plane.
        return volume[slice_index, :, :]

    raise ValueError(f"Unsupported plane: {plane}")


def extract_target_slice(
    volume: np.ndarray,
    plane: str,
    patient_center: np.ndarray,
    target_affine: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Extract the canonical slice closest to the source volume's
    physical center.

    The returned image is a canonical coordinate plane, so for
    oblique input it may not be exactly coplanar with the original
    source slice.
    """
    target_index = np.linalg.solve(
        target_affine[:3, :3],
        patient_center - target_affine[:3, 3],
    )

    axis = PLANE_TARGET_AXIS[plane]
    index = int(round(target_index[axis]))
    index = int(np.clip(index, 0, volume.shape[axis] - 1))

    if plane == "sagittal":
        image = volume[index, :, :]
    elif plane == "coronal":
        image = volume[:, index, :]
    else:
        image = volume[:, :, index]

    return image, index


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GEOMETRY-AWARE RESAMPLING VISUAL REVIEW")
    print("=" * 70)

    paths = DatasetPaths.from_root(DATASET_ROOT)
    metadata = MetadataReader(paths)
    metadata.validate()

    study_uid = metadata.get_study_uids()[0]

    selector = SeriesSelector(
        train_series_dir=paths.train_series_dir
    )
    selection = selector.select(study_uid, metadata.train_series)

    reader = DicomSeriesReader()
    volume_builder = DicomVolumeBuilder()
    resampler = VolumeResampler(interpolation_order=1)

    manifest_rows = []

    for plane in ("sagittal", "coronal", "axial"):
        print(f"\nProcessing {plane.upper()}...")

        selected = getattr(selection, plane)

        if selected is None:
            print("No series selected; skipping.")
            continue

        series = reader.read(selected.series_path)
        dicom_volume = volume_builder.build(series)
        source_volume = np.asarray(dicom_volume.volume)

        affine_result = calculate_voxel_to_patient_affine(series)

        source_affine = np.eye(4, dtype=np.float64)
        source_affine[:3, :3] = affine_result.matrix
        source_affine[:3, 3] = affine_result.origin

        grid = build_canonical_grid(
            source_shape=source_volume.shape,
            source_affine=source_affine,
            spacing=1.0,
        )

        resampled_volume = resampler.resample(
            volume=source_volume,
            source_affine=source_affine,
            target_grid=grid,
        )

        # Use the center of the central source slice as the
        # physical reference point.
        center_source_index = np.array([
            (source_volume.shape[0] - 1) / 2,
            (source_volume.shape[1] - 1) / 2,
            (source_volume.shape[2] - 1) / 2,
        ])

        center_patient = (
            source_affine[:3, :3] @ center_source_index
            + source_affine[:3, 3]
        )

        source_slice_index = source_volume.shape[0] // 2

        source_slice = extract_source_slice(
            source_volume,
            plane,
            source_slice_index,
        )

        target_slice, target_slice_index = extract_target_slice(
            resampled_volume,
            plane,
            center_patient,
            grid.affine,
        )

        # Save geometry for later reproducibility.
        plane_dir = OUTPUT_ROOT / plane
        plane_dir.mkdir(parents=True, exist_ok=True)

        np.save(plane_dir / "source_affine.npy", source_affine)
        np.save(plane_dir / "target_affine.npy", grid.affine)
        np.save(
            plane_dir / "target_shape.npy",
            np.asarray(grid.shape, dtype=np.int32),
        )

        # Display scaling is for visualization only.
        source_display = window_image(source_slice)
        target_display = window_image(target_slice)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(
            source_display,
            cmap="gray",
            origin="upper",
        )
        axes[0].set_title(
            f"Original {plane.capitalize()}\n"
            f"Shape: {source_slice.shape}"
        )
        axes[0].axis("off")

        axes[1].imshow(
            target_display,
            cmap="gray",
            origin="upper",
        )
        axes[1].set_title(
            f"Resampled canonical {plane.capitalize()}\n"
            f"Shape: {target_slice.shape}\n"
            f"Target slice index: {target_slice_index}"
        )
        axes[1].axis("off")

        fig.suptitle(
            f"Study: {study_uid}\n"
            f"Series: {series.series_instance_uid}",
            fontsize=10,
        )
        fig.tight_layout()

        image_path = plane_dir / "comparison.png"
        fig.savefig(image_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        manifest_rows.append({
            "study_uid": study_uid,
            "plane": plane,
            "series_uid": series.series_instance_uid,
            "source_shape": str(source_volume.shape),
            "target_shape": str(grid.shape),
            "source_slice_index": source_slice_index,
            "target_slice_index": target_slice_index,
            "spacing_mm": "1.0,1.0,1.0",
            "comparison_image": str(image_path),
        })

        print("Source shape:", source_volume.shape)
        print("Resampled shape:", grid.shape)
        print("Comparison saved:", image_path)

    manifest_path = OUTPUT_ROOT / "manifest.csv"

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        if manifest_rows:
            writer = csv.DictWriter(
                file,
                fieldnames=list(manifest_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

    print("\n" + "=" * 70)
    print("VISUAL REVIEW FILES CREATED")
    print("Output directory:", OUTPUT_ROOT.resolve())
    print("Manifest:", manifest_path.resolve())
    print("=" * 70)


if __name__ == "__main__":
    main()
