from pathlib import Path

from ish_knee.data.paths import DatasetPaths

from ish_knee.preprocessing import SeriesSelector
from ish_knee.preprocessing.dicom.dicom_reader import DicomSeriesReader
from ish_knee.data.metadata import MetadataReader

DATASET_ROOT = r"D:\RSNA knee abnormality detection 2026"

STUDY_UID = ("1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260")

def print_geometry(series_name, selected_series):

    print()
    print("=" * 70)
    print("SPACING ANALYZER TEST")
    print("=" * 70)

    if selected_series is None:
        print("no series selected")
        return

    reader = DicomSeriesReader()

    series = reader.read(Path(selected_series.series_path))

    ordered_slices = series.ordered_slices()

    print(f"Slcie count: {len(ordered_slices)}")
    print()

    previous_position = None

    for index, slice_ in enumerate(ordered_slices):

        position = slice_.image_position

        print(f"Slice {index + 1}")
        print(f"    Instance Number: {slice_.instance_number}")
        print(f"    Image position: {position}")

        if previous_position is not None and position is not None:

            import numpy as np

            first = np.array(previous_position, dtype=float)

            second = np.array(position, dtype=float)

            distance = np.linalg.norm(second - first)

            print(f"    Distance from previous: {distance:.6f} mm")   

            print()

            if position is not None:
                previous_position = position

def main():

    print()
    print("=" * 70)
    print("DICOM SLICE GEOMETRY TEST")
    print("=" * 70)   

    paths = DatasetPaths.from_root(DATASET_ROOT)

    metadata = MetadataReader(paths)

    series_metadata = metadata.get_study_series(STUDY_UID)

    selector = SeriesSelector(paths.train_series_dir)

    selection = selector.select(STUDY_UID, series_metadata)

    print()
    print("Study:")
    print(STUDY_UID)

    print_geometry("Sagittal", selection.sagittal)

    print_geometry("Coronal", selection.coronal)

    print_geometry("Axial", selection.axial)

if __name__ == "__main__":
    main()