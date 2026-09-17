from pathlib import Path
import sys

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

from ish_knee.data import(
    DatasetPaths,
    MetadataReader
)

from ish_knee.preprocessing import(
     SeriesSelector
)

from ish_knee.preprocessing.dicom import(
    DicomSeriesReader,
    DicomSeriesValidator
)

# --------------------------------------------------------------------------------------------------------------------
#  Configuration
# --------------------------------------------------------------------------------------------------------------------

DATASET_ROOT=(r"D:\RSNA knee abnormality detection 2026")


# --------------------------------------------------------------------------------------------------------------------
# Main test
# --------------------------------------------------------------------------------------------------------------------

def main():

    print("=" * 70)

    print("DICOM READER / SERIES VALIDATION TEST")

    print("=" * 70)

    # --------------------------------------------------------------------------------------------------------------------
    # Create Dataset Paths
    # --------------------------------------------------------------------------------------------------------------------

    paths = DatasetPaths.from_root(DATASET_ROOT)

    paths.validate()

    print("\nDataset root:")

    print(paths.root)

    # --------------------------------------------------------------------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------------------------------------------------------------------
  
    metadata = MetadataReader(paths)

    metadata.validate()

    print("\nMetadata loaded successfully")

    print(
        f"Train studies: "
        f"{len(metadata.train)}"
    )

    print(
        f"Train series: "
        f"{len((metadata.train_series))}"
    )

    # --------------------------------------------------------------------------------------------------------------------
    # Select one study
    # --------------------------------------------------------------------------------------------------------------------

    if metadata.train.empty:
        raise RuntimeError("Training metadata contains no studies")
    
    study_uid = str(metadata.train.iloc[0]["StudyInstanceUID"])

    print("\nSelected study:")

    print(study_uid)

    # --------------------------------------------------------------------------------------------------------------------
    # Show series metadata for the study
    # --------------------------------------------------------------------------------------------------------------------

    study_series = metadata.train_series[
        metadata.train_series["StudyInstanceUID"] == study_uid
    ].copy()

    print("\nSeries belonging to selected study:")

    print(
        study_series
    )

    if study_series is None:
        raise RuntimeError(
            f"No series metadata found for study "
            f"{study_uid}"
        )

    # --------------------------------------------------------------------------------------------------------------------
    # Select preferred series
    # --------------------------------------------------------------------------------------------------------------------

    selector = SeriesSelector(
        paths.train_series_dir
    )

    selection = selector.select(
        study_uid,
        metadata.train_series
    )

    reader = DicomSeriesReader()

    print()
    print("\n" + "=" * 70)
    print("SELECTED SERIES")
    print("=" * 70)

    _print_selected_series(
            "SAGITTAL",
            selection.sagittal
    )

    _print_selected_series(
            "CORONAL",
            selection.coronal
    )

    _print_selected_series(
            "AXIAL",
            selection.axial
    )

    # --------------------------------------------------------------------------------------------------------------------
    # Create DICOM reader and validator
    # --------------------------------------------------------------------------------------------------------------------  

    reader = DicomSeriesReader()

    validator = DicomSeriesValidator()

    # --------------------------------------------------------------------------------------------------------------------
    # Read and validate each selected plane
    # --------------------------------------------------------------------------------------------------------------------  

    selected_planes = [
        ("SAGITTAL", selection.sagittal),
        ("CORONAL", selection.coronal),
        ("AXIAL", selection.axial)
    ]

    for plane, selected in selected_planes:

        print("\n" +  "=" * 70)
        print(f"READING {plane} SERIES")
        print("= * 70")

        if selected is None:
            print(f"{plane}: No series was selected.")
            continue

        series_path = Path(selected.series_path)

        print(
            f"Series path:\n"
            f"{series_path}"
        )

        # --------------------------------------------------------------------------------------------------------------------
        # Verify directory
        # -------------------------------------------------------------------------------------------------------------------- 

        if not series_path.is_dir():
            print("\nERROR: Series directory does not exist.")
            continue

        # --------------------------------------------------------------------------------------------------------------------
        # Read DICOM series
        # -------------------------------------------------------------------------------------------------------------------- 

        dicom_series = reader.read(series_path)  

        print(
            f"\nImage count: "
            f"{dicom_series.image_count}"
        )    

        print(
            f"Image shape: "
            f"{dicom_series.image_shape}"
        )

        # --------------------------------------------------------------------------------------------------------------------
        # Display first slice
        # -------------------------------------------------------------------------------------------------------------------- 

        if dicom_series.slices:

            first_slice = (
                dicom_series.slices[0]
            )

            print("\nFirst slice:")

            print(
                f"  File: "
                f"{first_slice.path.name}"
            )
            
            print(
                f"  Instance Number: "
                f"{first_slice.instance_number}"
            )
            
            print(
                f"  Image position: "
                f"{first_slice.image_position}"
            )

            if first_slice.pixel_array is not None:
                
                print(
                    f"  Pixel dtype: "
                    f"{first_slice.pixel_array.dtype}"
                )
                 
                print(
                    f"  Pixel shape: "
                    f"{first_slice.pixel_array.shape}"
                )

        # --------------------------------------------------------------------------------------------------------------------
        # Order slices
        # -------------------------------------------------------------------------------------------------------------------- 

        ordered_slices = (
            dicom_series.ordered_slices()
        )

        print("\nSlice ordering:")

        print(
            f"  Original count: "
            f"{(len(dicom_series.slices))}"
        )

        print(
            f"  Ordered count: "
            f"{(len(ordered_slices))}"
        )

        if ordered_slices:

            first_ordered = (
                ordered_slices[0]
            )

            last_ordered = (
                ordered_slices[-1]
            )            

            print(
                "\n First ordered slice:"
            )

            print(
                f"  Instance: "
                f"{first_ordered.instance_number}"
            )

            print(
                f"  Position: "
                f"{first_ordered.image_position}"
            )

            print(
                "\n Last ordered slice:"
            )

            print(
                f"  Instance: "
                f"{last_ordered.instance_number}"
            )

            print(
                f"  Position: "
                f"{last_ordered.image_position}"
            )

        # --------------------------------------------------------------------------------------------------------------------
        # Validate series
        # -------------------------------------------------------------------------------------------------------------------- 

        validation = validator.validate(
            dicom_series
        )

        print("\nValidation:")

        if validation.is_valid:
            print(" PASS")
        else:
            print(" FAIL")

            for error in validation.errors:
                print(f"  ERROR: {error}")

        # --------------------------------------------------------------------------------------------------------------------
        # Validate series
        # -------------------------------------------------------------------------------------------------------------------- 

        print("\n" + "=" * 70)
        print("DICOM TEST COMPLETE")
        print("=" * 70)


# --------------------------------------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------------------------------------

def _print_selected_series(plane: str, selected)-> None:

    print(f"\n{plane}:")

    if selected is None:
        print(" No series is selected.")
        return

    print(
        f"  Series UID: "
        f"{selected.series_instance_uid}"
    )

    print(
        f"  Fluid Sensitive: "
        f"{selected.fluid_sensitive}"
    )

    print(
        f"  Fat suppression: "
        f"{selected.fat_suppression}"
    )

    print(
        f"  File count: "
        f"{selected.file_count}"
    )

    print(
        f"  Series Path: "
        f"{selected.series_path}"
    )

# --------------------------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()