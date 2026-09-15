from pathlib import Path
import sys

from ish_knee.preprocessing.Dicom.dicom_reader import DicomSeriesReader

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

from ish_knee.preprocessing.Dicom import(
    DicomSeriesReader
)

def main():

    dataset_root=(r"D:\RSNA knee abnormality detection 2026")

    paths = DatasetPaths.from_root(dataset_root)

    paths.validate()

    metadata = MetadataReader(paths)

    metadata.validate()

    study_uid = metadata.get_study_uids()[0]

    selector = SeriesSelector(
        paths.train_series_dir
    )

    # print("\nTRAIN COLUMNS:")
    # print(metadata.train.columns.tolist())

    # print("\nTRAIN SERIES COLUMNS:")
    # print(metadata.train_series.columns.tolist())

    # print("\nSELECTED STUDY SERIES:")
    # print(
    # metadata.train_series[
    #     metadata.train_series["StudyInstanceUID"] == study_uid
    #     ]
    # )

    selection = selector.select(
        study_uid,
        metadata.train_series
    )

    reader = DicomSeriesReader()

    print()
    print("=" * 70)
    print("DICOM READER TEST")
    print("=" * 70)

    if selection.sagittal is not None:

        print()
        print("Reading SAGITTAL series ...")

        sagittal = reader.read(selection.sagittal.series_path)

        print(
            f"Series path: "
            f"{sagittal.series_path}"
        )

        print(
            f"Image count: "
            f"{sagittal.slice_count}"
        )

        print(
            f"Image shape: "
            f"{sagittal.shape}"
        )

        first_slice = sagittal.slices[0]

        print(
            f"First slice: "
            f"{first_slice.path.name}"
        )

        print(
            f"Pixel dtype : "
            f"{first_slice.pixel_array.dtype}"
        )

        ordered_slices = dicom_series.ordered_slices()
if __name__ == "__main__":
    main()