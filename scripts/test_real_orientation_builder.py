from ish_knee.data import DatasetPaths, MetadataReader
from ish_knee.preprocessing import SeriesSelector
from ish_knee.preprocessing.orientation_transform_builder import OrientationTransformBuilder
from ish_knee.preprocessing.volume.orientation_analyzer import OrientationAnalyzer
from ish_knee.preprocessing.dicom import DicomSeriesReader

DATASET_ROOT = r"D:\RSNA knee abnormality detection 2026"

STUDY_UID = ("1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260")

def main():

    print()
    print("=" * 70)
    print("REAL RSNA ORIENTATION BUILDER TEST")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Load dataset metadata
    # -------------------------------------------------------------

    paths = DatasetPaths.from_root(DATASET_ROOT)

    metadata = MetadataReader(paths)

    train_series = metadata.train_series

    print()
    print("Metadata loaded")
    print(f"{train_series['SeriesInstanceUID'].nunique():,}")

    # -------------------------------------------------------------
    # 2. Get series belonging to the study
    # -------------------------------------------------------------

    study_series = metadata.get_study_series(STUDY_UID)

    print()
    print(f"Study: {STUDY_UID}")

    print()
    print(
        f"Series belonging to study: "
        f"{len(study_series)}"
    )

    # -------------------------------------------------------------
    # 3. Select the study's preferred series
    # -------------------------------------------------------------

    selector = SeriesSelector(paths.train_series_dir)

    selection = selector.select(STUDY_UID, study_series)

    # -------------------------------------------------------------
    # 4. Create processing objects
    # -------------------------------------------------------------

    reader = DicomSeriesReader()
    analyzer = OrientationAnalyzer()
    builder = OrientationTransformBuilder()

    # -------------------------------------------------------------
    # 5. Analyze each selected anatomical plane
    # -------------------------------------------------------------

    selected_series = [
        selection.sagittal,
        selection.coronal,
        selection.axial
    ]

    for selected in selected_series:

        if selected is None:
            print()
            print("Series not available")
            continue

    print()
    print("-" * 70)
    print(
        f"ANATOMICAL PLANE: "
        f"{selected.anatomical_plane}"
    )
    print("-" * 70)

    print()
    print("SERIES UID: ")
    print(selected.series_instance_uid)

    print()
    print("Series path: ")
    print(selected.series_path)

    # -------------------------------------------------------------
    # Read DICOM series
    # -------------------------------------------------------------

    series = reader.read(selected.series_path)

    print()
    print("Image count:")
    print(series.image_count)

    # -------------------------------------------------------------
    # Analyze DICOM orientation
    # -------------------------------------------------------------

    orientation = analyzer.analyze(series)

    print()
    print("Orientation information:")

    print("Row direction:   ", orientation.row_direction)

    print("Column direction:    ", orientation.column_direction)

    print("Slice normal:    ", orientation.slice_normal)

    print("Calculated plane:    ", orientation.anatomical_plane)

    print("Metadata plane:  ", orientation.anatomical_plane)

    # -------------------------------------------------------------
    # Verify anatomical plane
    # -------------------------------------------------------------

    if(orientation.anatomical_plane != selected.anatomical_plane):
        print()
        print("WARNING: Calculated plane does not match metadata")
    else:
        print()
        print("Anatomical plane: PASS")

    # -------------------------------------------------------------
    # Build orientation transformation
    # -------------------------------------------------------------

    transformation = builder.build(orientation)

    print()
    print("Generated orientation transform")

    print("Axis order:  ", transformation.axis_order)
 
    print("Flip axis 0:  ", transformation.flip_axis_0)
   
    print("Flip axis 1:  ", transformation.flip_axis_1)
     
    print("Flip axis 2:  ", transformation.flip_axis_2)

    print()
    print("Identity:  ", transformation.is_identity)

print()
print("=" * 70)
print("RAEL RSNA ORIENTATION BUILDER ETST COMPLETE")
print("=" * 70)
       
           
if __name__ == "__main__":
    main()