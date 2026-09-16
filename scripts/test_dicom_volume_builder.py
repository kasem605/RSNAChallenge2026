from pathlib import Path

from ish_knee.data import DatasetPaths, MetadataReader
from ish_knee.preprocessing import SeriesSelector
from ish_knee.preprocessing.Dicom.dicom_reader import (
    DicomSeriesReader,
)
from ish_knee.preprocessing.Dicom.dicom_volume_builder import (
    DicomVolumeBuilder,
)


def main():

    print("=" * 70)
    print("REAL DICOM VOLUME BUILDER TEST")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    dataset_root = Path(
        r"D:\RSNA knee abnormality detection 2026"
    )

    paths = DatasetPaths.from_root(dataset_root)

    print()
    print("Dataset root:")
    print(paths.root)

    # ------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------

    metadata = MetadataReader(paths)

    train_series = metadata.train_series

    print()
    print("Metadata loaded successfully")
    print(f"Train series: {len(train_series):,}")

    # ------------------------------------------------------------------
    # Select one study
    # ------------------------------------------------------------------

    study_uid = str(
        train_series.iloc[0]["StudyInstanceUID"]
    )

    print()
    print("Selected study:")
    print(study_uid)

    # ------------------------------------------------------------------
    # Select preferred MRI series
    # ------------------------------------------------------------------

    selector = SeriesSelector(
        paths.train_series_dir
    )

    selection = selector.select(
        study_uid,
        train_series,
    )

    # ------------------------------------------------------------------
    # DICOM reader and volume builder
    # ------------------------------------------------------------------

    reader = DicomSeriesReader()

    builder = DicomVolumeBuilder()

    # ------------------------------------------------------------------
    # Process selected planes
    # ------------------------------------------------------------------

    selected_planes = (
        ("sagittal", selection.sagittal),
        ("coronal", selection.coronal),
        ("axial", selection.axial),
    )

    for plane, selected_series in selected_planes:

        if selected_series is None:

            print()
            print(f"{plane.upper()}: No series selected")

            continue

        print()
        print("-" * 70)
        print(f"{plane.upper()} SERIES")
        print("-" * 70)

        print(
            "Series UID:     ",
            selected_series.series_instance_uid,
        )

        print(
            "Series path:    ",
            selected_series.series_path,
        )

        print(
            "Expected files: ",
            selected_series.file_count,
        )

        # --------------------------------------------------------------
        # Read real DICOM series
        # --------------------------------------------------------------

        series = reader.read(
            selected_series.series_path
        )

        print()
        print(
            "DICOM images:   ",
            series.image_count,
        )

        print(
            "Image shape:    ",
            series.image_shape,
        )

        # --------------------------------------------------------------
        # Build 3-D volume
        # --------------------------------------------------------------

        volume = builder.build(series)

        print()
        print("Volume created successfully")

        print(
            "Volume shape:   ",
            volume.shape,
        )

        print(
            "Slice count:    ",
            volume.slice_count,
        )

        print(
            "Data type:      ",
            volume.volume.dtype,
        )

        print(
            "Source path:    ",
            volume.source_path,
        )

        print(
            "Minimum value:  ",
            volume.volume.min(),
        )

        print(
            "Maximum value:  ",
            volume.volume.max(),
        )

        print(
            "Mean value:     ",
            f"{volume.volume.mean():.4f}",
        )

        # --------------------------------------------------------------
        # Validate
        # --------------------------------------------------------------

        assert volume.volume.ndim == 3

        assert volume.slice_count == series.image_count

        assert volume.shape[1:] == series.image_shape

        assert (
            volume.study_instance_uid
            == series.study_instance_uid
        )

        assert (
            volume.series_instance_uid
            == series.series_instance_uid
        )

        print()
        print("Assertions passed")

    print()
    print("=" * 70)
    print("REAL DICOM VOLUME BUILDER TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()