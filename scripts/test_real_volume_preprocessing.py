
from pathlib import Path
import numpy as np

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.preprocessing import SeriesSelector
from ish_knee.preprocessing.volume.preprocessing_config import PreprocessingConfig
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
from ish_knee.preprocessing.volume.volume_preprocessor import VolumePreprocessor
from ish_knee.dataset.knee_mri_processor import KneeMRIProcessor

DATASET_ROOT = r"D:\RSNA knee abnormality detection 2026"

STUDY_UID = (
    "1.2.826.0.1.3680043.8.498."
    "10004873229099053869093324292195817260"
)


def main():
    print("=" * 70)
    print("REAL MRI PREPROCESSING TEST")
    print("=" * 70)

    paths = DatasetPaths.from_root(DATASET_ROOT)
    metadata = MetadataReader(paths)
    selector = SeriesSelector(paths.train_series_dir)

    series_metadata = metadata.get_study_series(STUDY_UID)
    selection = selector.select(STUDY_UID, series_metadata)

    print("Study UID:", STUDY_UID)

    config = PreprocessingConfig(
        target_spacing=VoxelSpacing(1.0, 1.0, 1.0),
        target_shape=(128, 128, 128),
        normalize_intensity=True,
        padding_value=0.0,
    )

    preprocessor = VolumePreprocessor(config)
    processor = KneeMRIProcessor(preprocessor)

    # Check the selected series directly through the existing processor.
    for plane, selected_series in (
        ("Sagittal", selection.sagittal),
        ("Coronal", selection.coronal),
        ("Axial", selection.axial),
    ):
        print(f"\n{'-' * 70}")
        print(f"{plane.upper()} SERIES")
        print(f"{'-' * 70}")
        print("Series path:", selected_series.series_path)

        result = processor._process_series(selected_series)
        array = result.volume

        print("Final shape:", array.shape)
        print("Data type:", array.dtype)
        print("Finite values:", np.isfinite(array).all())
        print("Minimum:", float(array.min()))
        print("Maximum:", float(array.max()))

        assert array.shape == (128, 128, 128), (
            f"{plane}: unexpected shape {array.shape}"
        )
        assert np.isfinite(array).all(), (
            f"{plane}: NaN or infinite values found"
        )
        assert array.dtype == np.float32, (
            f"{plane}: unexpected dtype {array.dtype}"
        )

    print("\nREAL MRI PREPROCESSING TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
