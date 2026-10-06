from pathlib import Path
import numpy as np

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader

from ish_knee.preprocessing.series_selector import SeriesSelector

from ish_knee.preprocessing.volume.preprocessing_config import PreprocessingConfig
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
from ish_knee.preprocessing.volume.volume_preprocessor import VolumePreprocessor

from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder
from ish_knee.dataset.labeled_dataset_builder import LabeledDatasetBuilder
from ish_knee.dataset.knee_mri_processor import KneeMRIProcessor
from ish_knee.dataset.knee_dataset import KneeDataset

DATASET_ROOT = r"D:\RSNA knee abnormality detection 2026"

TARGET_SHAPE=(64, 256, 256)

def main() -> None:

    print("=" * 70)
    print("KNEE DATASET TEST")
    print("=" * 70)

    # ---------------------------------------------------------------------
    # Dataset paths
    # ---------------------------------------------------------------------

    paths = DatasetPaths.from_root(DATASET_ROOT)

    # ---------------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------------

    metadata = MetadataReader(paths)

    metadata.validate()

    print()
    print("Meatadata Loaded")
    print(f"Train studies: {len(metadata.train)}")
    print(f"Trains series: { len(metadata.train_series)}")

    # ---------------------------------------------------------------------
    # Series selector
    # ---------------------------------------------------------------------

    selector = SeriesSelector(paths.train_series_dir)

    # ---------------------------------------------------------------------
    # Dataset sample builder
    # ---------------------------------------------------------------------

    sample_builder = DatasetSampleBuilder(metadata=metadata, series_selector=selector)

    # ---------------------------------------------------------------------
    # Get fully labeled sample
    # ---------------------------------------------------------------------

    labeled_builder = LabeledDatasetBuilder(sample_builder=sample_builder)

    sample = labeled_builder.build_first()

    print()
    print("First labeled sample")
    print(f"Study UID: {sample.study_instance_uid}")
    print(f"Complete: {sample.study.is_complete}")

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    config = PreprocessingConfig(
        target_spacing=VoxelSpacing(
            1.0,
            1.0,
            1.0
        ),
        target_shape=TARGET_SHAPE
    )

    preprocessor = VolumePreprocessor(config)

    # ---------------------------------------------------------------------
    # MRI processor
    # ---------------------------------------------------------------------

    mri_processor = KneeMRIProcessor(preprocessor=preprocessor)

    # ---------------------------------------------------------------------
    # KneeDataset
    # ---------------------------------------------------------------------

    dataset = KneeDataset(
        samples=[sample],
        mri_processor=mri_processor
    )

    # ---------------------------------------------------------------------
    # Dataset tests
    # ---------------------------------------------------------------------

    assert len(dataset) == 1

    retrieved_sample = dataset.get_sample(0)

    assert retrieved_sample is sample

    assert (dataset.get_study_uid(0) == sample.study_instance_uid)

    print()
    print("Dataset count:", len(dataset))
    print("Datasaet UID:", dataset.get_study_uid(0))

    # ---------------------------------------------------------------------
    # Process MRI
    # ---------------------------------------------------------------------  

    mri_sample = dataset.get_mri(0)

    assert(mri_sample.study_instance_uid == sample.study_instance_uid)

    assert mri_sample.sagittal.shape == TARGET_SHAPE
    assert mri_sample.coronal.shape == TARGET_SHAPE
    assert mri_sample.axial.shape == TARGET_SHAPE

    assert mri_sample.sagittal.dtype == np.float32
    assert mri_sample.coronal.dtype == np.float32
    assert mri_sample.axial.dtype == np.float32   

    assert np.isfinite(mri_sample.sagittal).all()
    assert np.isfinite(mri_sample.coronal).all()
    assert np.isfinite(mri_sample.axial).all()

    print()
    print("Sagittal:",mri_sample.sagittal.shape) 
    print("Coronal:",mri_sample.coronal.shape) 
    print("Axial:",mri_sample.axial.shape) 

    print("=" * 70)
    print("KNEE DATASET TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    main()