from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")

TARGET_SHAPE = (64, 64, 64)
TARGET_SPACING = (1.0, 1.0, 1.0)
NUMBER_OF_STUDIES_TO_INSPECT = 8
NUMBER_OF_LABELS = 12


def add_project_root_to_python_path() -> None:
    """Allow imports when this script runs from the scripts folder."""
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def to_numpy(value: Any) -> np.ndarray:
    """Convert a PyTorch tensor or array-like value to NumPy."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def describe_volume(name: str, value: Any) -> int:
    """Print statistics and validate one preprocessed MRI volume."""
    array = to_numpy(value)
    issues = 0

    print(f"  {name} volume:")
    print(f"    Shape: {array.shape}")
    print(f"    Data type: {array.dtype}")

    if array.size == 0:
        print("    ERROR: Empty volume.")
        return 1

    finite_mask = np.isfinite(array)
    finite_count = int(finite_mask.sum())
    print(f"    Finite values: {finite_count:,}/{array.size:,}")
    print(f"    NaN/Inf values: {array.size - finite_count:,}")

    finite_values = None
    if finite_count > 0:
        finite_values = array[finite_mask].astype(np.float64, copy=False)
        print(f"    Minimum:  {finite_values.min():.6f}")
        print(f"    Maximum:  {finite_values.max():.6f}")
        print(f"    Mean:     {finite_values.mean():.6f}")
        print(f"    Std dev:  {finite_values.std():.6f}")

    if not finite_mask.all():
        print("    ERROR: Volume contains NaN or Inf.")
        issues += 1

    if tuple(array.shape) != TARGET_SHAPE:
        print(
            f"    WARNING: Expected shape {TARGET_SHAPE}, "
            f"but received {array.shape}."
        )
        issues += 1

    if finite_values is not None and finite_values.size and float(finite_values.std()) == 0.0:
        print("    WARNING: Volume has zero standard deviation.")
        issues += 1

    return issues


def describe_labels(value: Any) -> int:
    """Print statistics and validate one 12-element diagnostic target."""
    labels = to_numpy(value).reshape(-1)
    issues = 0

    print("  Diagnostic labels:")
    print(f"    Shape: {labels.shape}")
    print(f"    Values: {labels.tolist()}")
    finite = bool(np.isfinite(labels).all())
    binary = bool(np.isin(labels, [0, 1]).all())
    print(f"    All finite: {finite}")
    print(f"    All binary (0/1): {binary}")

    if labels.size != NUMBER_OF_LABELS:
        print(
            f"    ERROR: Expected {NUMBER_OF_LABELS} labels, "
            f"found {labels.size}."
        )
        issues += 1

    if not finite:
        print("    ERROR: Labels contain NaN or Inf.")
        issues += 1

    if not binary:
        print("    ERROR: Labels contain values other than 0 and 1.")
        issues += 1

    return issues


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
        from ish_knee.preprocessing.series_selector import SeriesSelector
        from ish_knee.model.study_split import StudySplit
        from ish_knee.model.model_dataset import ModelDataset
        from ish_knee.model.model_sample_builder import ModelSampleBuilder
        from ish_knee.model.pytorch_knee_dataset import PyTorchKneeDataset
        from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
    except Exception as exc:
        print(f"ERROR importing project classes: {type(exc).__name__}: {exc}")
        print(f"Project root: {PROJECT_ROOT}")
        return 1

    print("=" * 78)
    print("TRAINING TENSOR INSPECTION")
    print("=" * 78)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset root: {DATA_ROOT}")
    print(f"Expected volume shape: {TARGET_SHAPE}")
    print(f"Target voxel spacing: {TARGET_SPACING}")
    print()

    if not DATA_ROOT.exists():
        print(f"ERROR: Dataset root does not exist: {DATA_ROOT}")
        print("Check DATA_ROOT near the top of this script.")
        return 1

    try:
        paths = DatasetPaths.from_root(DATA_ROOT)
        metadata = MetadataReader(paths)
        metadata.validate()
        metadata.print_summary()

        series_selector = SeriesSelector(paths.train_series_dir)
        dataset_sample_builder = DatasetSampleBuilder(metadata, series_selector)
        labeled_dataset_builder = LabeledDatasetBuilder(dataset_sample_builder)
        labeled_samples = labeled_dataset_builder.build_all()

        if len(labeled_samples) < 2:
            raise ValueError(
                f"Expected at least two fully labeled studies; found {len(labeled_samples)}."
            )

        print(f"Fully labeled studies: {len(labeled_samples)}")
        training_samples, validation_samples = StudySplit.split(
            labeled_samples,
            validation_fraction=0.30,
            seed=42,
        )
        print(f"Training studies: {len(training_samples)}")
        print(f"Validation studies: {len(validation_samples)}")

        # Configure voxel spacing
        spacing = VoxelSpacing(
            spacing_axis_0=TARGET_SPACING[0],
            spacing_axis_1=TARGET_SPACING[1],
            spacing_axis_2=TARGET_SPACING[2],
        )
        # Match the preprocessing configuration used by the training pipeline.
        preprocessing_config = PreprocessingConfig(
            target_spacing=spacing,
            target_shape=TARGET_SHAPE,
            normalize_intensity=True,
            padding_value=0.0,
        )
        volume_preprocessor = VolumePreprocessor(preprocessing_config)
        mri_processor = KneeMRIProcessor(volume_preprocessor)
        model_sample_builder = ModelSampleBuilder(mri_processor)
        model_dataset = ModelDataset(training_samples, model_sample_builder)
        torch_dataset = PyTorchKneeDataset(model_dataset)
    except Exception as exc:
        print(f"ERROR building dataset: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    count = min(NUMBER_OF_STUDIES_TO_INSPECT, len(torch_dataset))
    if count == 0:
        print("ERROR: No training studies found.")
        return 1

    print()
    print(f"Inspecting the first {count} training studies.")
    print("No model training is performed.")
    print("No checkpoints are modified.")
    print()

    issues = 0

    for index in range(count):
        print("-" * 78)
        print(f"TRAINING STUDY {index + 1}/{count} (dataset index {index})")
        try:
            # IMPORTANT: PyTorchKneeDataset.__getitem__ returns FOUR items:
            # sagittal, coronal, axial, target. It does not return (inputs, target).
            sagittal, coronal, axial, target = torch_dataset[index]

            issues += describe_volume("Sagittal", sagittal)
            issues += describe_volume("Coronal", coronal)
            issues += describe_volume("Axial", axial)
            issues += describe_labels(target)
        except Exception as exc:
            print(f"ERROR inspecting study: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            issues += 1
        print()

    print("=" * 78)
    if issues:
        print(f"INSPECTION FINISHED WITH {issues} ISSUE(S).")
        print("Review the volume warnings and label errors above.")
        return 2

    print("INSPECTION FINISHED: No volume-shape, finite-value, or label-format issues detected.")
    print("Review the volume statistics for unexpectedly constant or unusually scaled inputs.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
