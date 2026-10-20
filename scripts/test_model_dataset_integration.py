from pathlib import Path

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader

from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder
from ish_knee.dataset.labeled_dataset_builder import LabeledDatasetBuilder

from ish_knee.preprocessing.volume.preprocessing_config import PreprocessingConfig
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
from ish_knee.preprocessing.volume.volume_preprocessor import VolumePreprocessor

from ish_knee.dataset.knee_mri_processor import KneeMRIProcessor

from ish_knee.model.model_sample_builder import ModelSampleBuilder
from ish_knee.model.model_dataset import ModelDataset

from ish_knee.preprocessing.series_selector import SeriesSelector

def main():

    print("=" * 70)
    print("MODEL DATASET INTEGRATION TEST")
    print("=" * 70)

    # ----------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------

    dataset_root = Path(
        r"D:\RSNA knee abnormality detection 2026"
    )

    paths = DatasetPaths.from_root(dataset_root)

    metadata = MetadataReader(paths)
    metadata.validate()

    series_selector = SeriesSelector(paths.train_series_dir)

    sample_builder = DatasetSampleBuilder(metadata, series_selector)

    labeled_builder = LabeledDatasetBuilder(sample_builder)

    samples = labeled_builder.build_all()

    print()
    print("labeled samples:", len(samples))

    assert len(samples) == 58

    # ----------------------------------------------------------
    # MRI Processing
    # ----------------------------------------------------------

    config = PreprocessingConfig(
        target_spacing=VoxelSpacing(
            spacing_axis_0=0,
            spacing_axis_1=0,
            spacing_axis_2=0
        ),
        target_shape=(64,64,64)
    )

    preprocessor = VolumePreprocessor(config)

    mri_processor = KneeMRIProcessor(preprocessor)

    # ----------------------------------------------------------
    # Model layer
    # ----------------------------------------------------------

    modele_sample_builder = ModelSampleBuilder(mri_processor)

    model_dataset = ModelDataset(samples=samples, sample_builder=modele_sample_builder)

    # ----------------------------------------------------------
    # Basic dataset checks
    # ----------------------------------------------------------

    assert len(model_dataset) == 58

    print("Model dataset length:", len(model_dataset))

    first_sample = model_dataset.get_sample(0)

    print("First study UID:", first_sample.study_instance_uid)

    # ----------------------------------------------------------
    # Do not process MRI volume yet
    # ----------------------------------------------------------

    print()
    print("ModelDataset structure verified")
    print("MRI volumes remain on-demand")

    print()
    print("MODEL DATASET INTEGRATION TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    main()
