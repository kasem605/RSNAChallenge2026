
from pathlib import Path

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader

from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder
from ish_knee.dataset.labeled_dataset_builder import LabeledDatasetBuilder
from ish_knee.dataset.knee_mri_processor import KneeMRIProcessor

from ish_knee.model.study_split import StudySplit
from ish_knee.model.model_dataset import ModelDataset
from ish_knee.model.model_sample_builder import ModelSampleBuilder

from ish_knee.preprocessing.volume.preprocessing_config import (
    PreprocessingConfig,
)
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
from ish_knee.preprocessing.volume.volume_preprocessor import (
    VolumePreprocessor,
)


def main() -> None:
    print("=" * 70)
    print("MODEL DATASET TRAIN / VALIDATION SPLIT TEST")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load metadata
    # --------------------------------------------------

    paths = DatasetPaths.from_root(
        Path(r"D:\RSNA knee abnormality detection 2026")
    )

    metadata = MetadataReader(paths)
    metadata.validate()

    # --------------------------------------------------
    # 2. Build fully labeled study samples
    # --------------------------------------------------

    series_selector = SeriesSelector(paths.train_series_dir)

    dataset_sample_builder = DatasetSampleBuilder(
        metadata=metadata,
        series_selector=series_selector,
    )

    labeled_dataset_builder = LabeledDatasetBuilder(
        dataset_sample_builder
    )

    samples = labeled_dataset_builder.build_all()

    print(f"Fully labeled studies: {len(samples)}")

    assert len(samples) == 58, (
        f"Expected 58 fully labeled studies; got {len(samples)}"
    )

    # --------------------------------------------------
    # 3. Split by study
    # --------------------------------------------------

    training_samples, validation_samples = StudySplit.split(
        samples,
        validation_fraction=0.20,
        seed=42,
    )

    print(f"Training studies:      {len(training_samples)}")
    print(f"Validation studies:    {len(validation_samples)}")

    assert len(training_samples) == 46
    assert len(validation_samples) == 12

    # --------------------------------------------------
    # 4. Verify no study overlap
    # --------------------------------------------------

    training_uids = {
        sample.study_instance_uid
        for sample in training_samples
    }

    validation_uids = {
        sample.study_instance_uid
        for sample in validation_samples
    }

    assert training_uids.isdisjoint(validation_uids), (
        "A study appears in both training and validation."
    )

    assert training_uids | validation_uids == {
        sample.study_instance_uid for sample in samples
    }

    # --------------------------------------------------
    # 5. Configure the model sample builder
    # --------------------------------------------------

    # These settings only initialize the pipeline here.
    # MRI preprocessing is not run by this split test.

    config = PreprocessingConfig(
        target_spacing=VoxelSpacing(
            spacing_axis_0=1.0,
            spacing_axis_1=1.0,
            spacing_axis_2=1.0,
        ),
        target_shape=(64, 64, 64),
    )

    preprocessor = VolumePreprocessor(config)
    mri_processor = KneeMRIProcessor(preprocessor)
    model_sample_builder = ModelSampleBuilder(mri_processor)

    # --------------------------------------------------
    # 6. Create separate model datasets
    # --------------------------------------------------

    training_dataset = ModelDataset(
        samples=training_samples,
        sample_builder=model_sample_builder,
    )

    validation_dataset = ModelDataset(
        samples=validation_samples,
        sample_builder=model_sample_builder,
    )

    assert len(training_dataset) == 46
    assert len(validation_dataset) == 12

    # --------------------------------------------------
    # 7. Verify study UID access
    # --------------------------------------------------

    for index in range(len(training_dataset)):
        assert training_dataset.get_study_uid(index)

    for index in range(len(validation_dataset)):
        assert validation_dataset.get_study_uid(index)

    print()
    print("Training ModelDataset:   PASSED")
    print("Validation ModelDataset: PASSED")
    print("Study overlap check:     PASSED")
    print("Study UID access:        PASSED")
    print()
    print("MODEL DATASET SPLIT TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
