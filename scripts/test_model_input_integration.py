from ish_knee.data import DatasetPaths
from ish_knee.data.metadata import MetadataReader

from ish_knee.preprocessing.series_selector import SeriesSelector

from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder
from ish_knee.dataset.knee_dataset import KneeDataset
from ish_knee.dataset.knee_mri_processor import KneeMRIProcessor

from ish_knee.model.model_input import ModelInput

from ish_knee.preprocessing.volume.preprocessing_config import PreprocessingConfig
from ish_knee.preprocessing.volume.volume_preprocessor import VolumePreprocessor
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing

def find_first_labeled_study(metadata: MetadataReader) -> str:

    LABEL_COLUMNS = (
            "ACL",
            "MCL",
            "Medial Meniscus",
            "Lateral Meniscus",
            "Medial OA",
            "Lateral OA",
            "PF OA",
            "Effusion",
            "Synovitis",
            "Baker's",
            "Contusion",
            "Fracture"
        )

    train = metadata.train

    labeled_rows = train[train[list(LABEL_COLUMNS)].notna().all(axis=1)]

    if labeled_rows.empty:
        raise ValueError("No labeled studies found in the training set")

    return str(labeled_rows.iloc[0]["StudyInstanceUID"])


def main() -> None:

    print("=" * 70)
    print("MODEL INPUT READ DATA INETEGRATION TEST")
    print("=" * 70)

    # -----------------------------------------------------------
    # Dataset paths
    # -----------------------------------------------------------

    DATASET_ROOT = r"D:\RSNA knee abnormality detection 2026"

    paths = DatasetPaths.from_root(DATASET_ROOT)

    # -----------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------
    
    metadata = MetadataReader(paths)

    metadata.validate()

    print()
    print("metadata loaded successfully")
    print(f"Train studies: {len(metadata.train)}")
    print(f"Train series: {len(metadata.train_series)}")

    # -----------------------------------------------------------
    # find fully labeled study
    # -----------------------------------------------------------

    study_uid = find_first_labeled_study(metadata)

    print()
    print("Selected labeled study:")
    print(study_uid)

    # -----------------------------------------------------------
    # Series selector
    # -----------------------------------------------------------

    series_selector = SeriesSelector(train_series_dir=paths.train_series_dir)

    # -----------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------

    sample_builder = DatasetSampleBuilder(metadata=metadata, series_selector=series_selector)

    dataset_sample = sample_builder.build(study_instance_uid=study_uid)

    print()
    print("DatsetSample created successfully")

    print("Study UID:")
    print(dataset_sample.study_instance_uid)

    print("Labels:")
    print(dataset_sample.labels.as_tuple)

    # -----------------------------------------------------------
    # Preprocessing configuration
    #
    # These values are only for this integration test
    # -----------------------------------------------------------

    preprocessing_config = PreprocessingConfig(
        target_spacing=VoxelSpacing(
            spacing_axis_0=1.0,
            spacing_axis_1=1.0,
            spacing_axis_2=1.0
        ),
        target_shape=(64, 64, 64)
    )
    preprocessor = VolumePreprocessor(config=preprocessing_config)

    # -----------------------------------------------------------
    # MRI processor
    # -----------------------------------------------------------

    mri_processor = KneeMRIProcessor(preprocessor=preprocessor)

    # -----------------------------------------------------------
    # KneeDataset
    # -----------------------------------------------------------    

    dataset = KneeDataset(samples=[dataset_sample], mri_processor=mri_processor)

    print()
    print("KneeDataset created")
    print(f"Dataset length: {len(dataset)}")

    # -----------------------------------------------------------
    # Process real MRI data
    # -----------------------------------------------------------

    mri_sample = dataset.get_mri(0)

    print()
    print("KNeeMRISample created successfully")

    print("Study UID:")
    print(mri_sample.study_instance_uid)

    print("Sagittal shape")
    print(mri_sample.sagittal_shape)
    
    print("Coronal shape")
    print(mri_sample.coronal_shape)
    
    print("Axial shape")
    print(mri_sample.axial_shape)
    
    if not mri_sample.is_3d:
        raise AssertionError("Processed MRI sample must contain three 3-D volumes")

    # -----------------------------------------------------------
    # Convert to ModelInput
    # -----------------------------------------------------------

    model_input = ModelInput.from_mri_sample(mri_sample)

    print()
    print("ModelInput created successfully")

    print("Study UID:")
    print(model_input.study_instance_uid)

    print("Sagittal Shape:")
    print(model_input.sagittal.shape)
    
    print("Coronal Shape:")
    print(model_input.coronal.shape)
    
    print("Axial Shape:")
    print(model_input.axial.shape)


    # ---------------------------------------------------------------------
    # Verify study identity
    # ---------------------------------------------------------------------

    if model_input.study_instance_uid != study_uid:
        raise AssertionError("SudyInstanceUID changed during processing")

    # ---------------------------------------------------------------------
    # Verify target shapes
    # ---------------------------------------------------------------------   

    expected_shape = (64, 64, 64)

    if model_input.sagittal.shape != expected_shape: 
        raise AssertionError(f"Unexpected sagittal shape: {model_input.sagittal.shape}") 

    if model_input.coronal.shape != expected_shape: 
        raise AssertionError(f"Unexpected coronal shape: {model_input.coronal.shape}") 

    if model_input.axial.shape != expected_shape: 
        raise AssertionError(f"Unexpected axial shape: {model_input.axial.shape}") 

    print()
    print("All ModelInput checks passed")

    print("=" * 70)
    print("MODEL INPUT READ DATA INETEGRATION TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    main()