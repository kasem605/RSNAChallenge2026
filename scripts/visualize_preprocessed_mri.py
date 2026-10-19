from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "mri_visual_check"

TARGET_SHAPE = (64, 64, 64)
TARGET_SPACING_VALUES = (1.0, 1.0, 1.0)


def add_project_root_to_python_path() -> None:
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def make_montage(volumes: list[tuple[str, np.ndarray]], output_path: Path) -> None:
    """Save 3 central-region slices for each of the three MRI orientations."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fractions = (0.25, 0.50, 0.75)

    for row, (orientation, volume) in enumerate(volumes):
        if tuple(volume.shape) != TARGET_SHAPE:
            raise ValueError(
                f"{orientation} volume has shape {volume.shape}; expected {TARGET_SHAPE}."
            )
        if not np.isfinite(volume).all():
            raise ValueError(f"{orientation} volume contains NaN or infinite values.")

        for col, fraction in enumerate(fractions):
            slice_index = min(int(round((volume.shape[0] - 1) * fraction)), volume.shape[0] - 1)
            image = volume[slice_index, :, :]
            axes[row, col].imshow(image, cmap="gray", origin="lower")
            axes[row, col].set_title(f"{orientation} — slice {slice_index}")
            axes[row, col].axis("off")

    fig.suptitle(
        "Preprocessed knee MRI — first training study\n"
        "Review anatomy visibility, cropping, and excessive padding",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    add_project_root_to_python_path()

    try:
        from ish_knee.data.paths import DatasetPaths
        from ish_knee.data.metadata import MetadataReader
        from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder
        from ish_knee.dataset.labeled_dataset_builder import LabeledDatasetBuilder
        from ish_knee.dataset.knee_mri_processor import KneeMRIProcessor
        from ish_knee.preprocessing.volume.preprocessing_config import PreprocessingConfig
        from ish_knee.preprocessing.volume.volume_preprocessor import VolumePreprocessor
        from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
        from ish_knee.preprocessing.series_selector import SeriesSelector
        from ish_knee.model.study_split import StudySplit
        from ish_knee.model.model_dataset import ModelDataset
        from ish_knee.model.model_sample_builder import ModelSampleBuilder
        from ish_knee.model.pytorch_knee_dataset import PyTorchKneeDataset
    except Exception as exc:
        print(f"ERROR importing project classes: {type(exc).__name__}: {exc}")
        print(f"Project root resolved to: {PROJECT_ROOT}")
        return 1

    print("=" * 72)
    print("PREPROCESSED MRI VISUAL CHECK")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset root : {DATA_ROOT}")
    print(f"Output image : {OUTPUT_DIR / 'training_study_1_mri_montage.png'}")
    print("This script does not train a model or modify checkpoints.")
    print("=" * 72)

    if not DATA_ROOT.exists():
        print(f"ERROR: Dataset root does not exist: {DATA_ROOT}")
        return 1

    try:
        paths = DatasetPaths.from_root(DATA_ROOT)
        metadata = MetadataReader(paths)
        metadata.validate()

        series_selector = SeriesSelector(paths.train_series_dir)
        sample_builder = DatasetSampleBuilder(metadata, series_selector)
        labeled_builder = LabeledDatasetBuilder(sample_builder)
        labeled_samples = labeled_builder.build_all()

        if not labeled_samples:
            raise ValueError("No fully labeled training studies were found.")

        training_samples, _ = StudySplit.split(
            labeled_samples,
            validation_fraction=0.30,
            seed=42,
        )

        target_spacing = VoxelSpacing(
            spacing_axis_0=TARGET_SPACING_VALUES[0],
            spacing_axis_1=TARGET_SPACING_VALUES[1],
            spacing_axis_2=TARGET_SPACING_VALUES[2],
        )
        preprocessing_config = PreprocessingConfig(
            target_spacing=target_spacing,
            target_shape=TARGET_SHAPE,
            normalize_intensity=True,
            padding_value=0.0,
        )
        volume_preprocessor = VolumePreprocessor(preprocessing_config)
        mri_processor = KneeMRIProcessor(volume_preprocessor)
        model_sample_builder = ModelSampleBuilder(mri_processor)
        model_dataset = ModelDataset(training_samples, model_sample_builder)
        torch_dataset = PyTorchKneeDataset(model_dataset)

        if len(torch_dataset) == 0:
            raise ValueError("The PyTorch training dataset is empty.")

        sagittal, coronal, axial, target = torch_dataset[0]
        volumes = [
            ("Sagittal", to_numpy(sagittal)),
            ("Coronal", to_numpy(coronal)),
            ("Axial", to_numpy(axial)),
        ]
        labels = to_numpy(target).reshape(-1)
        print(f"Selected first training study; label count: {labels.size}")
        for name, volume in volumes:
            print(
                f"{name}: shape={volume.shape}, min={volume.min():.4f}, "
                f"max={volume.max():.4f}, mean={volume.mean():.4f}, "
                f"std={volume.std():.4f}"
            )

        output_path = OUTPUT_DIR / "training_study_1_mri_montage.png"
        make_montage(volumes, output_path)
        print("\nVISUAL CHECK IMAGE CREATED")
        print(f"Open this file: {output_path}")
        print("Inspect whether the knee anatomy is visible and centered, and whether")
        print("the slices appear mostly empty or dominated by black padding.")
        return 0

    except Exception as exc:
        print(f"ERROR creating MRI montage: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
