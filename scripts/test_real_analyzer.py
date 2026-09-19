import sys
from pathlib import Path

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.preprocessing.dicom.dicom_reader import DicomSeriesReader
from ish_knee.preprocessing.volume.orientation_analyzer import OrientationAnalyzer

print("=" * 70)
print("REAL RSNA DICOM ORIENTATION TEST")
print("=" * 70)

# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

dataset_root = Path(r"D:\RSNA knee abnormality detection 2026")

paths = DatasetPaths.from_root(dataset_root)

metadata = MetadataReader(paths)

train_series = metadata.train_series

print()
print("Metadata loaded successfully")
print(f"Train series: {len(train_series)}")

# -----------------------------------------------------------------
# Select one real study
# -----------------------------------------------------------------

study_uid = str(train_series.iloc[0]["StudyInstanceUID"])

print()
print("Selected study:")
print(study_uid)

selector = SeriesSelector(paths.train_series_dir)

selection = selector.select(
    study_uid,
    train_series
)

# -----------------------------------------------------------------
# Analyze each available plane
# -----------------------------------------------------------------

reader = DicomSeriesReader()
analyzer = OrientationAnalyzer()

for plane in (
    "sagittal",
    "coronal",
    "axial"
):
    selected_series = getattr(
        selection,
        plane
    )

    if selected_series is None:
        print()
        print(f"{plane}: no series selected")
        continue

    print()
    print("-" * 70)
    print(f"{plane.upper()} SERIES")
    print("-" * 70)

    print(
        "Series UID:",
        selected_series.series_instance_uid
    )

    print(
        "Metadata plane:",
        selected_series.anatomical_plane
    )

    print(
        "Series path:",
        selected_series.series_path
    )

    # Read DICOM series

    series = reader.read(selected_series.series_path)

    print(
        "DICOM image count:",
        series.image_count
    )

    # Analyze orientation

    orientation = analyzer.analyze(series)

    print(
        "Row direction:",
        orientation.row_direction
    )

    print(
        "Column direction:",
        orientation.column_direction
    )

    print(
        "Slice normal:",
        orientation.slice_normal
    )

    print(
        "Calculated plane:",
        orientation.anatomical_plane
    )

    # Compare metadata against DICOM geometry

    if(orientation.anatomical_plane != selected_series.anatomical_plane):
        raise AssertionError(
            f"Orientation mismatch for {plane}: "
            f"metadata ="
            f"{selected_series.anatomical_plane}, "
            f"DICOM="
            f"{orientation.anatomical_plane}"
        )

    print()
    print("=" * 70)
    print("REAL RSNA ORIENTATION TEST PASSED")
    print("=" * 70)

        

