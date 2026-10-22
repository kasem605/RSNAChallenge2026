
from pathlib import Path
import sys

import numpy as np

# Allow running this script directly from the project root.
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
from ish_knee.preprocessing.volume.orientation_transformer import (
    OrientationTransformer,
)
from ish_knee.preprocessing.volume.orientation_transform_calculator import (
    OrientationTransformCalculator,
)
from ish_knee.preprocessing.volume.spacing_analyzer import SpacingAnalyzer


def main():
    # IMPORTANT:
    # Adjust the DatasetPaths construction below to match the way
    # your existing scripts initialize this dataclass.
    paths = DatasetPaths.from_root(
        Path(r"D:\RSNA knee abnormality detection 2026")
    )

    metadata = MetadataReader(paths)
    train = metadata.train

    labeled_columns = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA", "Effusion",
        "Synovitis", "Baker's", "Contusion", "Fracture",
    ]

    labeled = train[
        train[labeled_columns].notna().all(axis=1)
    ]

    print(f"Fully labeled studies: {len(labeled)}")
    selected_studies = labeled["StudyInstanceUID"].head(3).tolist()

    selector = SeriesSelector(
        train_series_dir=paths.train_series_dir
    )
    reader = DicomSeriesReader()
    volume_builder = DicomVolumeBuilder()
    orientation_analyzer = OrientationAnalyzer()
    transform_calculator = OrientationTransformCalculator()
    transformer = OrientationTransformer()
    spacing_analyzer = SpacingAnalyzer()

    for study_index, study_uid in enumerate(selected_studies, start=1):
        print("\n" + "=" * 72)
        print(f"Study {study_index}: {study_uid}")

        selection = selector.select(study_uid, metadata.train_series)

        for plane in ("sagittal", "coronal", "axial"):
            selected = getattr(selection, plane, None)

            if selected is None:
                print(f"\n{plane}: no series selected")
                continue

            print(f"\n--- {plane.upper()} ---")

            series = reader.read(selected.series_path)
            volume = volume_builder.build(series)
            info = orientation_analyzer.analyze(series)
            spacing = spacing_analyzer.analyze(series)

            transform = transform_calculator.calculate(info)
            transformed = transformer.transform(volume, transform)

            source_spacing = spacing.as_tuple
            output_spacing = tuple(
                source_spacing[axis]
                for axis in transform.axis_order
            )

            print("Anatomical plane:", info.anatomical_plane)
            print("Row direction:", info.row_direction)
            print("Column direction:", info.column_direction)
            print("Slice normal:", info.slice_normal)
            print("Original shape:", volume.volume.shape)
            print("Original spacing (mm):", source_spacing)
            print("Proposed axis order:", transform.axis_order)
            print(
                "Proposed flips:",
                (
                    transform.flip_axis_0,
                    transform.flip_axis_1,
                    transform.flip_axis_2,
                ),
            )
            print("Transformed shape:", transformed.volume.shape)
            print("Transformed spacing (mm):", output_spacing)

            expected_shape = tuple(
                volume.volume.shape[axis]
                for axis in transform.axis_order
            )

            assert transformed.volume.shape == expected_shape
            assert len(output_spacing) == 3
            assert all(
                np.isfinite(value) and value > 0
                for value in output_spacing
            )

    print("\nORIENTATION AUDIT COMPLETE")
    print("No dataset files were modified.")


if __name__ == "__main__":
    main()
