
from pathlib import Path

import numpy as np

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.preprocessing.dicom.dicom_reader import (
    DicomSeriesReader,
)
from ish_knee.preprocessing.dicom.dicom_volume_builder import (
    DicomVolumeBuilder,
)
from ish_knee.preprocessing.geometry.voxel_to_patient_affine import (
    calculate_voxel_to_patient_affine,
)
from ish_knee.preprocessing.geometry.canonical_grid import (
    build_canonical_grid,
)
from ish_knee.preprocessing.geometry.volume_resampler import (
    VolumeResampler,
)


DATASET_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")


def main():
    print("=" * 70)
    print("REAL DICOM VOLUME RESAMPLING TEST")
    print("=" * 70)

    paths = DatasetPaths.from_root(DATASET_ROOT)
    metadata = MetadataReader(paths)
    metadata.validate()

    study_uid = metadata.get_study_uids()[0]

    selector = SeriesSelector(
        train_series_dir=paths.train_series_dir
    )
    selection = selector.select(study_uid, metadata.train_series)

    reader = DicomSeriesReader()
    volume_builder = DicomVolumeBuilder()
    resampler = VolumeResampler(interpolation_order=1)

    for plane in ("sagittal", "coronal", "axial"):
        print(f"\n{'-' * 70}")
        print(f"PLANE: {plane.upper()}")
        print("-" * 70)

        selected = getattr(selection, plane)

        if selected is None:
            print("No series selected; skipping.")
            continue

        series = reader.read(selected.series_path)

        # Build the source 3D volume.
        dicom_volume = volume_builder.build(series)
        source_volume = np.asarray(dicom_volume.volume)

        # Calculate source voxel -> patient LPS affine.
        affine_result = calculate_voxel_to_patient_affine(series)

        source_affine = np.eye(4, dtype=np.float64)
        source_affine[:3, :3] = affine_result.matrix
        source_affine[:3, 3] = affine_result.origin

        # Construct an axis-aligned 1 mm LPS output grid.
        grid = build_canonical_grid(
            source_shape=source_volume.shape,
            source_affine=source_affine,
            spacing=1.0,
        )

        # Resample using the physical coordinate transforms.
        result = resampler.resample(
            volume=source_volume,
            source_affine=source_affine,
            target_grid=grid,
        )

        print("Source shape:", source_volume.shape)
        print("Target shape:", result.shape)
        print("Target spacing:", grid.spacing)
        print("Source dtype:", source_volume.dtype)
        print("Output dtype:", result.dtype)
        print(
            "Source intensity range:",
            float(np.min(source_volume)),
            "to",
            float(np.max(source_volume)),
        )
        print(
            "Output intensity range:",
            float(np.min(result)),
            "to",
            float(np.max(result)),
        )
        print("All output values finite:", np.isfinite(result).all())
        print("Output C-contiguous:", result.flags.c_contiguous)

        assert result.shape == grid.shape
        assert result.dtype == np.float32
        assert np.isfinite(result).all()
        assert result.flags.c_contiguous

        print("PASS: Resampling completed")

    print("\n" + "=" * 70)
    print("REAL DICOM RESAMPLING TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
