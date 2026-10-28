
from pathlib import Path
import sys

import numpy as np

# Ensure the project root is on Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.preprocessing.dicom.dicom_reader import DicomSeriesReader
from ish_knee.preprocessing.dicom.dicom_volume_builder import (
    DicomVolumeBuilder,
)
from ish_knee.preprocessing.volume.orientation_analyzer import (
    OrientationAnalyzer,
)
from ish_knee.preprocessing.orientation.orientation_transform_calculator import (
    OrientationTransformCalculator,
)
from ish_knee.preprocessing.volume.orientation_transformer import (
    OrientationTransformer,
)


DATASET_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")

STUDY_UIDS = [
    "1.2.826.0.1.3680043.8.498.10095687747295410396510538520594649149",
    "1.2.826.0.1.3680043.8.498.10170898615867673028696505248839028269",
    "1.2.826.0.1.3680043.8.498.10306159113324811538703788080836752052",
]


def main():
    print("=" * 78)
    print("ANATOMICAL ORIENTATION TEST")
    print("=" * 78)

    paths = DatasetPaths.from_root(DATASET_ROOT)

    metadata = MetadataReader(paths)
    train_series_df = metadata.train_series

    selector = SeriesSelector(
        train_series_dir=paths.train_series_dir
    )
    reader = DicomSeriesReader()
    volume_builder = DicomVolumeBuilder()
    analyzer = OrientationAnalyzer()
    calculator = OrientationTransformCalculator()
    transformer = OrientationTransformer()

    failures = []

    for study_uid in STUDY_UIDS:
        print(f"\n{'=' * 78}")
        print(f"Study: {study_uid}")
        print("=" * 78)

        selection = selector.select(study_uid, train_series_df)

        for plane_name in ("sagittal", "coronal", "axial"):
            selected = getattr(selection, plane_name, None)

            if selected is None:
                print(f"\n{plane_name.upper()}: No selected series")
                continue

            print(f"\n--- {plane_name.upper()} ---")

            try:
                series = reader.read(selected.series_path)
                volume = volume_builder.build(series)

                orientation = analyzer.analyze(series)
                transform = calculator.calculate(series)

                transformed = transformer.transform(
                    volume,
                    transform,
                )

                print(f"Series UID: {series.series_instance_uid}")
                print(f"Slice count: {len(series.slices)}")
                print(f"Detected plane: {orientation.anatomical_plane}")

                print("Row direction:   ", orientation.row_direction)
                print("Column direction:", orientation.column_direction)
                print("Slice normal:    ", orientation.slice_normal)

                print("Original shape:   ", volume.volume.shape)
                print("Axis order:       ", transform.axis_order)
                print(
                    "Flips:            ",
                    (
                        transform.flip_axis_0,
                        transform.flip_axis_1,
                        transform.flip_axis_2,
                    ),
                )
                print("Transformed shape:", transformed.volume.shape)

                print(
                    "Finite voxel values:",
                    bool(np.isfinite(transformed.volume).all()),
                )

                print(
                    "Memory contiguous:",
                    bool(transformed.volume.flags["C_CONTIGUOUS"]),
                )

                if orientation.anatomical_plane != plane_name:
                    print(
                        "WARNING: Selected plane differs from "
                        "the plane inferred from DICOM geometry."
                    )

            except Exception as exc:
                failures.append(
                    (study_uid, plane_name, str(exc))
                )
                print(f"FAILED: {type(exc).__name__}: {exc}")

    print(f"\n{'=' * 78}")
    print("TEST SUMMARY")
    print("=" * 78)

    if failures:
        print(f"Failed series: {len(failures)}")
        for study_uid, plane_name, error in failures:
            print(f"- {study_uid} | {plane_name}: {error}")
    else:
        print("No processing exceptions occurred.")

    print(
        "\nNOTE: Successful execution does not prove that anatomical "
        "directions or physical-coordinate metadata are correct."
    )


if __name__ == "__main__":
    main()
