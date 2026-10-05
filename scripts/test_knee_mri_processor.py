from pathlib import Path
import numpy as np

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader

from ish_knee.preprocessing.series_selector import SeriesSelector

from ish_knee.preprocessing.volume.preprocessing_config import PreprocessingConfig
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing

from ish_knee.preprocessing.volume.volume_preprocessor import VolumePreprocessor

from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder
from ish_knee.dataset.knee_mri_processor import KneeMRIProcessor
from ish_knee.dataset.labeled_dataset_builder import LabeledDatasetBuilder

DATASET_ROOT = r"D:\RSNA knee abnormality detection 2026"

STUDY_UID = ("1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260")

TARGET_SHAPE = (64, 256, 256)

def main() -> None:

    print("=" * 70)
    print("KNEE MRI PROCESSOR TEST")
    print("=" * 70)

    # -------------------------------------------------------------------
    # Dataset paths
    # -------------------------------------------------------------------

    paths = DatasetPaths.from_root(DATASET_ROOT)

    # -------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------

    metadata = MetadataReader(paths)

    metadata.validate()

    print()
    print("Metadata loaded")
    print(f"Train studies: {len(metadata.train)}")
    print(f"Train series: {len(metadata.train_series)}")

    # -------------------------------------------------------------------
    # Series selctor
    # -------------------------------------------------------------------

    selector = SeriesSelector(paths.train_series_dir)

    # -------------------------------------------------------------------
    # Dataset sample builder
    # -------------------------------------------------------------------

    sample_builder = DatasetSampleBuilder(metadata=metadata, series_selector=selector)

    labeled_builder = LabeledDatasetBuilder(sample_builder=sample_builder)

    sample = labeled_builder.build_first()

    print()
    print("Fully labeled dataset sample created")
    print(f"Study UID: {sample.study_instance_uid}")
    print(f"Complete: {sample.study.is_complete}")
    print(f"Labels: {sample.labels.as_tuple}")

    # -------------------------------------------------------------------
    # Preprocessing configuration
    # -------------------------------------------------------------------  

    config = PreprocessingConfig(
        target_spacing=VoxelSpacing(
            1.0,
            1.0,
            1.0
        ),
        target_shape=TARGET_SHAPE
        )

    preprocessor = VolumePreprocessor(config)

    # -------------------------------------------------------------------
    # MRI processor
    # -------------------------------------------------------------------  

    processor = KneeMRIProcessor(preprocessor=preprocessor)

    # -------------------------------------------------------------------
    # Process one study
    # ------------------------------------------------------------------- 

    mri_sample = processor.process(sample) 

    # -------------------------------------------------------------------
    # Validate result
    # -------------------------------------------------------------------  

    assert mri_sample.study_instance_uid == sample.study_instance_uid

    assert mri_sample.sagittal.shape == TARGET_SHAPE
    assert mri_sample.coronal.shape == TARGET_SHAPE
    assert mri_sample.axial.shape == TARGET_SHAPE

    print()
    print("Sagittal:", mri_sample.sagittal.shape )
    print("Coronal:", mri_sample.coronal.shape )
    print("Axial:", mri_sample.axial.shape )

    print()
    print("Dtype", mri_sample.sagittal.dtype )

    print()
    print("KNEE MRI PROCESSOR TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    main()
