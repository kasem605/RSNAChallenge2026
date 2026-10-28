
from pathlib import Path

import numpy as np

from ish_knee.data.paths import DatasetPaths
from ish_knee.preprocessing import SeriesSelector
from ish_knee.preprocessing.dicom.dicom_reader import DicomSeriesReader
from ish_knee.data.metadata import MetadataReader

from ish_knee.preprocessing.volume.spacing_analyzer import SpacingAnalyzer

DATASET_ROOT = r"D:\RSNA knee abnormality detection 2026"

STUDY_UID = (
    "1.2.826.0.1.3680043.8.498."
    "10004873229099053869093324292195817260"
)


def print_geometry(series_name, selected_series):

    print()
    print("=" * 70)
    print(f"{series_name.upper()} SLICE GEOMETRY")
    print("=" * 70)

    if selected_series is None:
        print("No series selected.")
        return

    reader = DicomSeriesReader()
    series = reader.read(Path(selected_series.series_path))

    spacing_analyzer = SpacingAnalyzer()
    spacing = spacing_analyzer.analyze(series)

    print(f"SpacingAnalyzer result: {spacing.as_tuple}")

    ordered_slices = series.ordered_slices()

    print(f"Slice count: {len(ordered_slices)}")

    if len(ordered_slices) < 2:
        print("Not enough slices to calculate spacing.")
        return

    orientation = ordered_slices[0].image_orientation

    if orientation is None:
        print("Missing image orientation; cannot calculate slice normal.")
        return

    orientation = np.asarray(orientation, dtype=float)

    row_direction = orientation[:3]
    column_direction = orientation[3:]
    normal = np.cross(row_direction, column_direction)

    normal_length = np.linalg.norm(normal)

    if not np.isfinite(normal_length) or normal_length < 1e-8:
        print("Invalid slice orientation.")
        return

    normal = normal / normal_length

    previous_position = None
    projected_positions = []

    for index, slice_ in enumerate(ordered_slices):

        position = slice_.image_position

        print(f"\nSlice {index + 1}")
        print(f"    Instance Number: {slice_.instance_number}")
        print(f"    Image position: {position}")

        if position is None:
            print("    Missing image position.")
            previous_position = None
            continue

        position_array = np.asarray(position, dtype=float)
        projected_position = float(np.dot(position_array, normal))
        projected_positions.append(projected_position)

        print(f"    Position along normal: {projected_position:.6f} mm")

        if previous_position is not None:
            distance = projected_position - previous_position
            print(f"    Signed spacing from previous: {distance:.6f} mm")

        previous_position = projected_position

    if len(projected_positions) < 2:
        print("\nInsufficient valid positions to assess spacing.")
        return

    differences = np.diff(projected_positions)
    absolute_spacing = np.abs(differences)

    print("\nSpacing summary")
    print(f"    Minimum: {absolute_spacing.min():.6f} mm")
    print(f"    Median:  {np.median(absolute_spacing):.6f} mm")
    print(f"    Maximum: {absolute_spacing.max():.6f} mm")

    if np.any(differences < -1e-4):
        print("WARNING: Slices are not in ascending spatial order.")
    else:
        print("Spatial ordering: PASS")

    if np.any(absolute_spacing < 1e-4):
        print("WARNING: Duplicate or near-duplicate slice positions.")

    median_spacing = np.median(absolute_spacing)

    if median_spacing > 0:
        irregular = (
            np.abs(absolute_spacing - median_spacing)
            > 0.10 * median_spacing
        )

        if np.any(irregular):
            print("WARNING: Spacing varies by more than 10% from the median.")
        else:
            print("Spacing consistency: PASS (within 10% of median)")


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

    print("\nStudy:")
    print(STUDY_UID)

    print_geometry("Sagittal", selection.sagittal)
    print_geometry("Coronal", selection.coronal)
    print_geometry("Axial", selection.axial)


if __name__ == "__main__":
    main()
