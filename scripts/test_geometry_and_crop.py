
from pathlib import Path

import numpy as np

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.preprocessing import SeriesSelector
from ish_knee.preprocessing.dicom.dicom_reader import DicomSeriesReader
from ish_knee.preprocessing.dicom.dicom_volume_builder import DicomVolumeBuilder
from ish_knee.preprocessing.volume.spacing_analyzer import SpacingAnalyzer
from ish_knee.preprocessing.volume.preprocessing_config import PreprocessingConfig
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
from ish_knee.preprocessing.volume.volume_preprocessor import VolumePreprocessor


DATASET_ROOT = r"D:\RSNA knee abnormality detection 2026"

STUDY_UID = (
    "1.2.826.0.1.3680043.8.498."
    "10004873229099053869093324292195817260"
)

TARGET_SHAPE = (128, 128, 128)
TARGET_SPACING = np.array([1.0, 1.0, 1.0])


def physical_extent(shape, spacing):
    """Approximate physical extent in mm using voxel counts."""
    return np.asarray(shape, dtype=float) * np.asarray(spacing, dtype=float)


def print_extent(label, shape, spacing):
    extent = physical_extent(shape, spacing)
    print(
        f"{label}: "
        f"{extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} mm"
    )


def main():
    print("=" * 70)
    print("REAL MRI GEOMETRY AND CROP VALIDATION")
    print("=" * 70)
    print(f"Study UID: {STUDY_UID}")

    paths = DatasetPaths.from_root(DATASET_ROOT)
    metadata = MetadataReader(paths)
    selector = SeriesSelector(paths.train_series_dir)

    series_metadata = metadata.get_study_series(STUDY_UID)
    selection = selector.select(STUDY_UID, series_metadata)

    reader = DicomSeriesReader()
    builder = DicomVolumeBuilder()
    spacing_analyzer = SpacingAnalyzer()

    config = PreprocessingConfig(
        target_spacing=VoxelSpacing(1.0, 1.0, 1.0),
        target_shape=TARGET_SHAPE,
        normalize_intensity=True,
        padding_value=0.0,
    )
    preprocessor = VolumePreprocessor(config)

    selected_planes = (
        ("Sagittal", selection.sagittal),
        ("Coronal", selection.coronal),
        ("Axial", selection.axial),
    )

    for plane, selected_series in selected_planes:
        print("\n" + "-" * 70)
        print(f"{plane.upper()} SERIES")
        print("-" * 70)

        series = reader.read(Path(selected_series.series_path))
        volume = builder.build(series)
        spacing = spacing_analyzer.analyze(series)

        original = volume.volume
        original_shape = np.asarray(original.shape)
        original_spacing = np.asarray(spacing.as_tuple, dtype=float)

        print(f"Original voxel shape: {tuple(original_shape)}")
        print(f"Original spacing (mm): {original_spacing}")
        print_extent(
            "Approx. original extent",
            original_shape,
            original_spacing,
        )


        # The production method performs resampling, cropping/padding,
        # and normalization. Use it to verify the final result.
        processed = preprocessor.preprocess(volume, spacing)
        final = processed.volume

        print(f"Final voxel shape: {final.shape}")
        print(f"Final dtype: {final.dtype}")
        print(f"Final intensity range: {final.min():.4f} to {final.max():.4f}")
        print(f"All values finite: {np.isfinite(final).all()}")

        assert final.shape == TARGET_SHAPE
        assert final.dtype == np.float32
        assert np.isfinite(final).all()

    print("\n" + "=" * 70)
    print("GEOMETRY AND CROP VALIDATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
