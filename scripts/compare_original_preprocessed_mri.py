from __future__ import annotations

import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "mri_visual_check"
TARGET_SHAPE = (128, 128, 128)
TARGET_SPACING_VALUES = (1.0, 1.0, 1.0)
BACKGROUND_THRESHOLD = 0.02


def add_project_root_to_python_path() -> None:
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def read_original_dicom_volume(series_path: str) -> tuple[np.ndarray, list[Path]]:
    """Read a selected series for visual comparison; does not replace the project reader."""
    folder = Path(series_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"DICOM series folder does not exist: {folder}")

    datasets = []
    for path in folder.iterdir():
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), force=True)
            if getattr(ds, "PixelData", None) is None:
                continue
            pixels = ds.pixel_array
            if pixels.ndim != 2:
                continue
            datasets.append((path, ds, pixels.astype(np.float32)))
        except Exception:
            continue

    if not datasets:
        raise ValueError(f"No readable 2D DICOM images found in: {folder}")

    first_ds = datasets[0][1]
    orientation = getattr(first_ds, "ImageOrientationPatient", None)
    can_sort_by_position = orientation is not None and all(
        getattr(ds, "ImagePositionPatient", None) is not None
        for _, ds, _ in datasets
    )
    if can_sort_by_position:
        row_direction = np.asarray(orientation[:3], dtype=np.float64)
        column_direction = np.asarray(orientation[3:6], dtype=np.float64)
        normal = np.cross(row_direction, column_direction)
        datasets.sort(
            key=lambda item: float(
                np.dot(np.asarray(item[1].ImagePositionPatient, dtype=np.float64), normal)
            )
        )
    else:
        datasets.sort(
            key=lambda item: (float(getattr(item[1], "InstanceNumber", 0)), item[0].name)
        )

    slices = []
    for _, ds, pixels in datasets:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        slices.append(pixels * slope + intercept)

    return np.stack(slices, axis=0), [item[0] for item in datasets]


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def describe_volume(name: str, volume: np.ndarray) -> None:
    volume = np.asarray(volume)
    if not np.isfinite(volume).all():
        raise ValueError(f"{name} contains NaN or infinite values.")

    total = volume.size
    zero_pct = 100.0 * float(np.count_nonzero(volume == 0)) / total
    low_pct = 100.0 * float(np.count_nonzero(volume <= BACKGROUND_THRESHOLD)) / total
    foreground = volume > BACKGROUND_THRESHOLD

    if foreground.any():
        coords = np.argwhere(foreground)
        mins, maxs = coords.min(axis=0), coords.max(axis=0)
        bbox_shape = maxs - mins + 1
        bbox_pct = 100.0 * float(np.prod(bbox_shape)) / total
        bbox = (
            f"min={tuple(int(v) for v in mins)}, max={tuple(int(v) for v in maxs)}, "
            f"shape={tuple(int(v) for v in bbox_shape)}, "
            f"bbox volume={bbox_pct:.1f}%"
        )
    else:
        bbox = "no voxels above background threshold"

    print(
        f"{name}: shape={volume.shape}, min={volume.min():.5f}, "
        f"max={volume.max():.5f}, mean={volume.mean():.5f}, std={volume.std():.5f}"
    )
    print(f"  exact zeros={zero_pct:.1f}%; values <= {BACKGROUND_THRESHOLD:.3f}={low_pct:.1f}%")
    print(f"  approximate anatomy bounding box: {bbox}")


def make_comparison_montage(
    comparisons: list[tuple[str, np.ndarray, np.ndarray]], output_path: Path
) -> None:
    fig, axes = plt.subplots(len(comparisons), 2, figsize=(10, 14))
    if len(comparisons) == 1:
        axes = np.asarray([axes])

    for row, (name, original, processed) in enumerate(comparisons):
        oi = original[original.shape[0] // 2]
        pi = processed[processed.shape[0] // 2]
        axes[row, 0].imshow(normalize_for_display(oi), cmap="gray", origin="lower")
        axes[row, 0].set_title(
            f"{name}: original DICOM\nslice {original.shape[0] // 2}, shape {oi.shape}"
        )
        axes[row, 0].axis("off")
        axes[row, 1].imshow(normalize_for_display(pi), cmap="gray", origin="lower")
        axes[row, 1].set_title(
            f"{name}: preprocessed\nslice {processed.shape[0] // 2}, shape {pi.shape}"
        )
        axes[row, 1].axis("off")

    fig.suptitle(
        "Original vs. preprocessed knee MRI\n"
        "Qualitative comparison; slice positions and axes may differ",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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
    print("ORIGINAL / PREPROCESSED MRI DIAGNOSTIC")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset root : {DATA_ROOT}")
    print("No model training is performed; checkpoints are not touched.")
    print("=" * 72)

    if not DATA_ROOT.exists():
        print(f"ERROR: Dataset root does not exist: {DATA_ROOT}")
        return 1

    try:
        paths = DatasetPaths.from_root(DATA_ROOT)
        metadata = MetadataReader(paths)
        metadata.validate()
        selector = SeriesSelector(paths.train_series_dir)
        sample_builder = DatasetSampleBuilder(metadata, selector)
        labeled_samples = LabeledDatasetBuilder(sample_builder).build_all()
        if not labeled_samples:
            raise ValueError("No fully labeled training studies were found.")

        training_samples, _ = StudySplit.split(
            labeled_samples, validation_fraction=0.30, seed=42
        )
        if not training_samples:
            raise ValueError("Training split is empty.")

        target_spacing = VoxelSpacing(
            spacing_axis_0=TARGET_SPACING_VALUES[0],
            spacing_axis_1=TARGET_SPACING_VALUES[1],
            spacing_axis_2=TARGET_SPACING_VALUES[2],
        )
        config = PreprocessingConfig(
            target_spacing=target_spacing,
            target_shape=TARGET_SHAPE,
            normalize_intensity=True,
            padding_value=0.0,
        )
        mri_processor = KneeMRIProcessor(VolumePreprocessor(config))
        model_sample_builder = ModelSampleBuilder(mri_processor)
        model_dataset = ModelDataset(training_samples, model_sample_builder)
        torch_dataset = PyTorchKneeDataset(model_dataset)
        if len(torch_dataset) == 0:
            raise ValueError("The PyTorch training dataset is empty.")

        sample = training_samples[0]
        study = sample.study
        print(f"\nStudyInstanceUID: {study.study_instance_uid}")
        print(f"Training samples: {len(training_samples)}")

        series_by_plane = [
            ("Sagittal", study.sagittal),
            ("Coronal", study.coronal),
            ("Axial", study.axial),
        ]
        for name, series in series_by_plane:
            if series is None:
                raise ValueError(f"Selected study has no {name} series.")
            print(
                f"{name} SeriesInstanceUID: {series.series_instance_uid}\n"
                f"  Folder: {series.series_path}\n"
                f"  Files: {series.file_count}"
            )

        sagittal, coronal, axial, target = torch_dataset[0]
        processed = [
            ("Sagittal", to_numpy(sagittal).squeeze()),
            ("Coronal", to_numpy(coronal).squeeze()),
            ("Axial", to_numpy(axial).squeeze()),
        ]
        print(f"\nLabel count: {to_numpy(target).reshape(-1).size}")

        originals = {}
        for name, series in series_by_plane:
            volume, source_files = read_original_dicom_volume(series.series_path)
            originals[name] = volume
            print(f"\nOriginal {name}: slices={len(source_files)}, shape={volume.shape}")

        print("\nPREPROCESSED VOLUME METRICS")
        for name, volume in processed:
            if tuple(volume.shape) != TARGET_SHAPE:
                raise ValueError(
                    f"{name} processed shape is {volume.shape}; expected {TARGET_SHAPE}."
                )
            describe_volume(f"Processed {name}", volume)

        print("\nORIGINAL DICOM VOLUME METRICS")
        for name, _ in series_by_plane:
            describe_volume(f"Original {name}", originals[name])

        comparisons = [(name, originals[name], volume) for name, volume in processed]
        output_path = OUTPUT_DIR / "original_vs_preprocessed_mri.png"
        make_comparison_montage(comparisons, output_path)

        print("\nDIAGNOSTIC COMPLETED")
        print(f"Comparison montage: {output_path}")
        print(
            "Note: original and processed slices may differ in orientation and location. "
            "Use this montage for qualitative anatomy/cropping checks, not pixel alignment."
        )
        print(f"Approximate anatomy threshold: {BACKGROUND_THRESHOLD:.3f}")
        return 0

    except Exception as exc:
        print(f"ERROR running MRI diagnostic: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
