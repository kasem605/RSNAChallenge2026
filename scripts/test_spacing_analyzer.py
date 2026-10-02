
from ish_knee.data import (DatasetPaths, MetadataReader)
from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.preprocessing.dicom.dicom_reader import DicomSeriesReader
from ish_knee.preprocessing.volume.spacing_analyzer import SpacingAnalyzer

DATASET_ROOT = r"D:\RSNA knee abnormality detection 2026"

STUDY_UID = ("1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260")

def main():

    print()
    print("=" * 70)
    print("SPACING ANALYZER TEST")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Load metadata
    # -----------------------------------------------------------------------

    paths = DatasetPaths.from_root(DATASET_ROOT)

    metadata = MetadataReader(paths)

    study_series = metadata.get_study_series(STUDY_UID)

    print()
    print("Study:")
    print(STUDY_UID)

    # -----------------------------------------------------------------------
    # 2. Select preferred series
    # -----------------------------------------------------------------------

    selector = SeriesSelector(paths.train_series_dir)

    selection = selector.select(STUDY_UID, study_series)

    # -----------------------------------------------------------------------
    # 3. Create reader and analyzer
    # -----------------------------------------------------------------------

    reader = DicomSeriesReader()

    analyzer = SpacingAnalyzer()

    selected_series = [
        selection.sagittal,
        selection.coronal,
        selection.axial
    ]

    # -----------------------------------------------------------------------
    # 4. Analyze each series
    # -----------------------------------------------------------------------

    for selected in selected_series:
        if selected is None:
            continue

        print()
        print("-" * 70)
        print(f"PLANE: {selected.anatomical_plane}")
        print("-" * 70)

        series = reader.read(selected.series_path)

        spacing = analyzer.analyze(series)

        print()
        print("Voxel spacing:")
        print(spacing)

        print()
        print("Spacing tuple:")
        print(spacing.as_tuple)

        # --------------------------------------------------------------------
        # Basic validation
        # --------------------------------------------------------------------

        assert(spacing.spacing_axis_0 > 0)
        
        assert(spacing.spacing_axis_1 > 0)
        
        assert(spacing.spacing_axis_2 > 0)

        print()
        print("Positive spacing values: PASS")

    print()
    print("=" * 70)
    print("REAL RSNA VOXEL SPACING TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    main()

