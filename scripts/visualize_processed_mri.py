
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.preprocessing import SeriesSelector
from ish_knee.preprocessing.volume.preprocessing_config import PreprocessingConfig
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
from ish_knee.preprocessing.volume.volume_preprocessor import VolumePreprocessor
from ish_knee.dataset.knee_mri_processor import KneeMRIProcessor


DATASET_ROOT = r"D:\RSNA knee abnormality detection 2026"

STUDY_UID = (
    "1.2.826.0.1.3680043.8.498."
    "10004873229099053869093324292195817260"
)

OUTPUT_DIR = Path("outputs") / "mri_visualization"
TARGET_SHAPE = (128, 128, 128)


def save_volume_slices(volume, plane, output_dir):
    """Save three neighboring slices from the processed 3D volume."""
    array = np.asarray(volume)
    axis = 0
    center = array.shape[axis] // 2

    indices = sorted({
        max(0, center - 8),
        center,
        min(array.shape[axis] - 1, center + 8),
    })

    fig, axes = plt.subplots(1, len(indices), figsize=(12, 4))

    if len(indices) == 1:
        axes = [axes]

    for ax, index in zip(axes, indices):
        image = np.take(array, index, axis=axis)
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Slice {index}")
        ax.axis("off")

    fig.suptitle(f"{plane} — processed MRI")
    fig.tight_layout()

    output_path = output_dir / f"{plane.lower()}_processed.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path.resolve()}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = DatasetPaths.from_root(DATASET_ROOT)
    metadata = MetadataReader(paths)
    selector = SeriesSelector(paths.train_series_dir)

    series_metadata = metadata.get_study_series(STUDY_UID)
    selection = selector.select(STUDY_UID, series_metadata)

    config = PreprocessingConfig(
        target_spacing=VoxelSpacing(1.0, 1.0, 1.0),
        target_shape=TARGET_SHAPE,
        normalize_intensity=True,
        padding_value=0.0,
    )

    processor = KneeMRIProcessor(VolumePreprocessor(config))

    selected_planes = (
        ("Sagittal", selection.sagittal),
        ("Coronal", selection.coronal),
        ("Axial", selection.axial),
    )

    for plane, selected_series in selected_planes:
        result = processor._process_series(selected_series)
        volume = result.volume

        print(f"{plane}: shape={volume.shape}, dtype={volume.dtype}")

        assert volume.shape == TARGET_SHAPE
        save_volume_slices(volume, plane, OUTPUT_DIR)

    print("\nVisualization complete.")
    print(f"Images are in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
