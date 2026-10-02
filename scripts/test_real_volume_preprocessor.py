from pathlib import Path

import numpy as np

from ish_knee.data import DatasetPaths, MetadataReader
from ish_knee.preprocessing import SeriesSelector
from ish_knee.preprocessing.dicom.dicom_reader import DicomSeriesReader
from ish_knee.preprocessing.volume.preprocessing_config import PreprocessingConfig
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
from ish_knee.preprocessing.volume.volume_preprocessor import VolumePreprocessor
from ish_knee.preprocessing.volume.volume_preprocessor import VolumePreprocessor
from ish_knee.preprocessing.volume.spacing_analyzer import SpacingAnalyzer
from ish_knee.preprocessing.dicom.dicom_volume_builder import DicomVolumeBuilder

DATASET_ROOT= Path(r"D:\RSNA knee abnormality detection 2026")

STUDY_UID = ("1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260")

TARGET_SHAPE = (
    64,
    256,
    256
)

def process_plane(
     plane_name,
     selected_series,
     reader,
     builder,
     spacing_analyzer,
     preprocessor   
):
    print()
    print("=" * 70)
    print(f"No {plane_name.upper()} REAL VOLUME PREPROCESSING")
    print("=" * 70)

    if selected_series is None:
        raise RuntimeError(f"No {plane_name} series selected")

    series_path = Path(selected_series.series_path)

    print()
    print("Series path:")
    print(series_path)

    # ----------------------------------------------------------------------
    # Read DICOM series
    # ----------------------------------------------------------------------

    series = reader.read(series_path)

    print()
    print("Slice count:")
    print(series.image_count)

    assert series.image_count > 0

    print("DICOM series: PASS")

    # ----------------------------------------------------------------------
    # Build volume
    # ----------------------------------------------------------------------    

    volume = builder.build(series)

    print()
    print("Original volume shape:")
    print(volume.volume.shape)

    assert volume.volume.ndim == 3

    print("Volume construction: PASS")

    # ----------------------------------------------------------------------
    # Analyze voxel spacing
    # ----------------------------------------------------------------------      

    spacing = spacing_analyzer.analyze(series)

    print()
    print("Current voxel spacing:")
    print(spacing)

    assert all(
        value > 0
        for value in spacing.as_tuple
    )

    print("Voxel spacing: PASS")

    # ----------------------------------------------------------------------
    # Preprocess
    # ----------------------------------------------------------------------  

    result = preprocessor.preprocess(
        volume=volume,
        current_spacing = spacing
    )

    print()
    print("Final volume shape:")
    print(result.volume.shape)

    # ----------------------------------------------------------------------
    # Verify target shape
    # ----------------------------------------------------------------------  

    assert result.volume.shape == TARGET_SHAPE

    print("Target shape: PASS")

    # ----------------------------------------------------------------------
    # Verify dtype
    # ----------------------------------------------------------------------  

    assert np.issubdtype(result.volume.dtype, np.floating)

    print("Floating-point output: PASS")

    # ----------------------------------------------------------------------
    # Verify normalization
    # ----------------------------------------------------------------------  

    minimum = np.min(result.volume)
    maximum = np.max(result.volume)
    
    print()
    print("Final intensity minimum:")
    print(minimum)

    print("Final intensity maximum:")
    print(maximum)

    assert minimum >= 0.0
    assert maximum <= 1.0

    print("Intensity normalization: PASS")

    # ----------------------------------------------------------------
    # Verify contiguous memory
    # ----------------------------------------------------------------

    assert result.volume.flags["C_CONTIGUOUS"]

    print("Contiguous memory: PASS")

    # ----------------------------------------------------------------
    # Verify finite values
    # ----------------------------------------------------------------

    assert np.isfinite(result.volume).all()

    print("Finite values: PASS")

    # ----------------------------------------------------------------
    # Verify metadata
    # ----------------------------------------------------------------

    assert result.study_instance_uid == STUDY_UID
    assert result.series_instance_uid == selected_series.series_instance_uid

    print("Metadata preservation: PASS")

    return result

def main():

    print("=" * 70)
    print("REAL RSNA VOLUME PREPROCESSOR TEST")
    print("=" * 70)

    # ----------------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------------    

    paths = DatasetPaths.from_root(DATASET_ROOT)

    metadata = MetadataReader(paths)

    metadata.validate()

    print()
    print("Metadata validation: PASS")

    # ----------------------------------------------------------------
    # Study metadata
    # ----------------------------------------------------------------

    series_metadata = metadata.get_study_series(STUDY_UID)

    assert not series_metadata.empty

    print("Study metadata: PASS")

    # ----------------------------------------------------------------
    # Select series
    # ----------------------------------------------------------------

    selector = SeriesSelector(paths.train_series_dir)

    selection = selector.select(STUDY_UID, series_metadata)

    # ----------------------------------------------------------------
    # Preprocessing configuration
    # ----------------------------------------------------------------

    config = PreprocessingConfig(
        target_spacing=VoxelSpacing(
            spacing_axis_0=1.0,
            spacing_axis_1=1.0,
            spacing_axis_2=1.0,            
        ),
        target_shape=TARGET_SHAPE,
        normalize_intensity=True,
        padding_value=0.0
    )

    print()
    print("Target shape:")
    print(config.target_shape)

    # ----------------------------------------------------------------
    # Components
    # ----------------------------------------------------------------

    reader = DicomSeriesReader()

    builder = DicomVolumeBuilder()

    spacing_analyzer = SpacingAnalyzer()

    preprocessor = VolumePreprocessor(config)

    # ----------------------------------------------------------------
    # Process all three planes
    # ----------------------------------------------------------------

    sagittal = process_plane(
        "sagittal",
        selection.sagittal,
        reader,
        builder,
        spacing_analyzer,
        preprocessor
    )

    coronal = process_plane(
        "coronal",
        selection.coronal,
        reader,
        builder,
        spacing_analyzer,
        preprocessor
    )

    axial = process_plane(
        "axial",
        selection.axial,
        reader,
        builder,
        spacing_analyzer,
        preprocessor
    )

    # ----------------------------------------------------------------
    # Final verification
    # ----------------------------------------------------------------

    assert sagittal.volume.shape == TARGET_SHAPE
    assert coronal.volume.shape == TARGET_SHAPE
    assert axial.volume.shape == TARGET_SHAPE   

    print()
    print("=" * 70)
    print("ALL THREE PLANES PREPROCESSED SUCCESSFULLY")
    print("=" * 70)

    print()
    print("Sagittal:", sagittal.volume.shape)
    print("Coronal :", coronal.volume.shape)
    print("Axial   :", axial.volume.shape)

    print("=" * 70)
    print("REAL RSNA VOLUME PREPROCESSOR TEST PASSED")
    print("=" * 70)


     
if __name__ == "__main__":
    main()