
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np


# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")

TARGET_SHAPE = (128, 128, 128)
TARGET_SPACING_VALUES = (1.0, 1.0, 1.0)

VALIDATION_FRACTION = 0.30
RANDOM_SEED = 42
EXPECTED_LABEL_COUNT = 12


def add_project_root_to_python_path() -> None:
    """Make the project's src directory importable."""
    src_path = PROJECT_ROOT / "src"

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def to_numpy(value) -> np.ndarray:
    """Convert a tensor or array to a NumPy array."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def validate_volume(name: str, value) -> list[str]:
    """Validate a single processed MRI volume."""
    errors = []
    volume = to_numpy(value).squeeze()

    if volume.shape != TARGET_SHAPE:
        errors.append(
            f"{name}: expected shape {TARGET_SHAPE}, "
            f"received {volume.shape}"
        )
        return errors

    if not np.issubdtype(volume.dtype, np.number):
        errors.append(f"{name}: volume is not numeric")
        return errors

    if not np.isfinite(volume).all():
        errors.append(f"{name}: contains NaN or infinite values")
        return errors

    if volume.size == 0:
        errors.append(f"{name}: volume is empty")

    return errors


def validate_target(target) -> list[str]:
    """Validate the twelve target labels without modifying them."""
    errors = []
    labels = to_numpy(target).reshape(-1)

    if labels.size != EXPECTED_LABEL_COUNT:
        errors.append(
            f"Expected {EXPECTED_LABEL_COUNT} labels, "
            f"received {labels.size}"
        )
        return errors

    if not np.issubdtype(labels.dtype, np.number):
        errors.append("Target labels are not numeric")
        return errors

    if not np.isfinite(labels).all():
        errors.append("Target labels contain NaN or infinite values")
        return errors

    # This project uses binary labels for fully labeled samples.
    # Do not replace missing labels with zero.
    if not np.isin(labels, [0, 1]).all():
        errors.append(
            "Target contains values other than 0 and 1: "
            f"{np.unique(labels).tolist()}"
        )

    return errors


def main() -> int:
    add_project_root_to_python_path()

    try:
        from ish_knee.data.paths import DatasetPaths
        from ish_knee.data.metadata import MetadataReader
        from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder
        from ish_knee.dataset.labeled_dataset_builder import LabeledDatasetBuilder
        from ish_knee.dataset.knee_mri_processor import KneeMRIProcessor
        from ish_knee.preprocessing.volume.preprocessing_config import (
            PreprocessingConfig,
        )
        from ish_knee.preprocessing.volume.volume_preprocessor import (
            VolumePreprocessor,
        )
        from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
        from ish_knee.preprocessing.series_selector import SeriesSelector
        from ish_knee.model.study_split import StudySplit
        from ish_knee.model.model_dataset import ModelDataset
        from ish_knee.model.model_sample_builder import ModelSampleBuilder
        from ish_knee.model.pytorch_knee_dataset import PyTorchKneeDataset

    except Exception as exc:
        print(
            f"ERROR importing project classes: "
            f"{type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return 1

    print("=" * 72)
    print("PREPROCESSED DATASET VALIDATION")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset root : {DATA_ROOT}")
    print(f"Target shape : {TARGET_SHAPE}")
    print(f"Target spacing: {TARGET_SPACING_VALUES}")
    print("No model training; no checkpoints are read or modified.")
    print("=" * 72)

    if not DATA_ROOT.exists():
        print(f"ERROR: Dataset root does not exist: {DATA_ROOT}")
        return 1

    started_at = time.perf_counter()

    passed = 0
    failed = 0
    failure_details = []

    try:
        # -------------------------------------------------
        # 1. Load metadata and build fully labeled samples
        # -------------------------------------------------
        print("\n[1/4] Loading dataset metadata...")

        paths = DatasetPaths.from_root(DATA_ROOT)
        metadata = MetadataReader(paths)
        metadata.validate()

        selector = SeriesSelector(paths.train_series_dir)
        sample_builder = DatasetSampleBuilder(metadata, selector)

        labeled_samples = LabeledDatasetBuilder(
            sample_builder
        ).build_all()

        if not labeled_samples:
            raise ValueError(
                "No fully labeled training studies were found."
            )

        training_samples, validation_samples = StudySplit.split(
            labeled_samples,
            validation_fraction=VALIDATION_FRACTION,
            seed=RANDOM_SEED,
        )

        if not training_samples:
            raise ValueError("Training split is empty.")

        print(f"Fully labeled samples: {len(labeled_samples)}")
        print(f"Training samples:      {len(training_samples)}")
        print(f"Validation samples:    {len(validation_samples)}")

        # -------------------------------------------------
        # 2. Configure the existing preprocessing pipeline
        # -------------------------------------------------
        print("\n[2/4] Configuring MRI preprocessing...")

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

        mri_processor = KneeMRIProcessor(
            VolumePreprocessor(config)
        )

        model_sample_builder = ModelSampleBuilder(mri_processor)
        model_dataset = ModelDataset(
            training_samples,
            model_sample_builder,
        )
        torch_dataset = PyTorchKneeDataset(model_dataset)

        if len(torch_dataset) != len(training_samples):
            raise ValueError(
                "Dataset length does not match the training split."
            )

        # -------------------------------------------------
        # 3. Validate every training sample
        # -------------------------------------------------
        print("\n[3/4] Validating training samples...")
        print("This may take some time on a CPU.\n")

        for index in range(len(torch_dataset)):
            sample_started = time.perf_counter()

            try:
                sagittal, coronal, axial, target = torch_dataset[index]

                errors = []

                errors.extend(
                    validate_volume("Sagittal", sagittal)
                )
                errors.extend(
                    validate_volume("Coronal", coronal)
                )
                errors.extend(
                    validate_volume("Axial", axial)
                )
                errors.extend(validate_target(target))

                elapsed = time.perf_counter() - sample_started

                if errors:
                    failed += 1

                    study_uid = (
                        training_samples[index]
                        .study.study_instance_uid
                    )

                    failure_details.append(
                        (index, study_uid, errors)
                    )

                    print(
                        f"[FAIL] Sample {index + 1}/"
                        f"{len(torch_dataset)} "
                        f"({elapsed:.1f}s)"
                    )

                    for error in errors:
                        print(f"       - {error}")

                else:
                    passed += 1

                    print(
                        f"[PASS] Sample {index + 1}/"
                        f"{len(torch_dataset)} "
                        f"({elapsed:.1f}s)"
                    )

                # Release references before processing the next sample.
                del sagittal, coronal, axial, target

            except Exception as exc:
                failed += 1

                study_uid = (
                    training_samples[index]
                    .study.study_instance_uid
                )

                failure_details.append(
                    (
                        index,
                        study_uid,
                        [f"{type(exc).__name__}: {exc}"],
                    )
                )

                print(
                    f"[ERROR] Sample {index + 1}/"
                    f"{len(torch_dataset)}: "
                    f"{type(exc).__name__}: {exc}"
                )

        # -------------------------------------------------
        # 4. Report results
        # -------------------------------------------------
        elapsed_total = time.perf_counter() - started_at

        print("\n[4/4] Validation summary")
        print("=" * 72)
        print(f"Training samples checked: {len(torch_dataset)}")
        print(f"Passed:                   {passed}")
        print(f"Failed:                   {failed}")
        print(f"Total elapsed time:       {elapsed_total / 60:.2f} minutes")

        if passed:
            print(
                f"Average time per passing sample: "
                f"{elapsed_total / max(len(torch_dataset), 1):.2f} seconds"
            )

        if failure_details:
            print("\nFailed sample details:")

            for index, study_uid, errors in failure_details:
                print(f"\nSample index: {index}")
                print(f"Study UID:    {study_uid}")

                for error in errors:
                    print(f"  - {error}")

        print("\n" + "=" * 72)

        if failed == 0:
            print("DATASET VALIDATION PASSED")
            print(
                "All training samples produced correctly shaped, "
                "finite volumes and valid binary targets."
            )
            return 0

        print("DATASET VALIDATION FAILED")
        print("Review the errors above before training.")
        return 1

    except Exception as exc:
        print(
            f"\nVALIDATION COULD NOT COMPLETE: "
            f"{type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
