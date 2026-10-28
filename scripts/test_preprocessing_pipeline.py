
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
from ish_knee.preprocessing.volume.intensity_normalizer import (
    IntensityNormalizer,
)
from ish_knee.preprocessing.volume.volume_resizer import (
    VolumeResizer,
    VolumeResizeConfig,
)
from ish_knee.preprocessing.geometry.resize_geometry_transformer import (
    ResizeGeometryTransformer,
)

DATASET_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")
OUTPUT_ROOT = Path("outputs/preprocessing_pipeline_test")

TARGET_SHAPE = (96, 160, 160)


def main():
    print("=" * 70)
    print("PREPROCESSING PIPELINE INTEGRATION TEST")
    print("=" * 70)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. Load metadata and select a study.
    paths = DatasetPaths.from_root(DATASET_ROOT)
    metadata = MetadataReader(paths)
    metadata.validate()

    study_uid = metadata.get_study_uids()[0]

    selector = SeriesSelector(
        train_series_dir=paths.train_series_dir
    )
    selection = selector.select(
        study_uid,
        metadata.train_series,
    )

    # Initialize existing components.
    reader = DicomSeriesReader()
    volume_builder = DicomVolumeBuilder()
    volume_resampler = VolumeResampler(
        interpolation_order=1
    )
    intensity_normalizer = IntensityNormalizer()
    volume_resizer = VolumeResizer(
        VolumeResizeConfig(target_shape=TARGET_SHAPE)
    )

    results = []

    for plane in ("sagittal", "coronal", "axial"):
        print(f"\nProcessing {plane.upper()}...")

        selected = getattr(selection, plane)

        if selected is None:
            print("No series selected; skipping.")
            continue

        # 2. Read DICOM series.
        series = reader.read(selected.series_path)

        # 3. Build source volume.
        source = volume_builder.build(series)

        print("Source shape:", source.volume.shape)

        # 4. Calculate voxel-to-patient geometry.
        affine_result = calculate_voxel_to_patient_affine(series)

        source_affine = np.eye(4, dtype=np.float64)
        source_affine[:3, :3] = affine_result.matrix
        source_affine[:3, 3] = affine_result.origin

        # 5. Build canonical physical grid.
        grid = build_canonical_grid(
            source_shape=source.volume.shape,
            source_affine=source_affine,
            spacing=1.0,
        )

        # 6. Resample into canonical physical space.
        canonical_array = volume_resampler.resample(
            volume=source.volume,
            source_affine=source_affine,
            target_grid=grid,
        )

        # Preserve DicomVolume metadata after resampling.
        from ish_knee.preprocessing.dicom.dicom_volume import (
            DicomVolume,
        )

        canonical_volume = DicomVolume(
            study_instance_uid=source.study_instance_uid,
            series_instance_uid=source.series_instance_uid,
            volume=canonical_array,
            source_path=source.source_path,
        )

        print("Canonical shape:", canonical_volume.volume.shape)

        # 7. Normalize intensities to [0, 1].
        normalized_volume = intensity_normalizer.normalize(
            canonical_volume
        )

        normalized = normalized_volume.volume

        assert np.isfinite(normalized).all()
        assert normalized.dtype == np.float32
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

        print("Normalized intensity range:",
              float(normalized.min()),
              "to",
              float(normalized.max()))

        # 8. Resize to fixed model dimensions.
        resized = volume_resizer.resize(normalized)

        assert resized.shape == TARGET_SHAPE
        assert resized.dtype == np.float32
        assert resized.flags.c_contiguous
        assert np.isfinite(resized).all()

        
        # 9. Calculate geometry for the resized volume.
        resized_geometry = ResizeGeometryTransformer.transform(
            source_shape=normalized_volume.volume.shape,
            source_affine=grid.affine,
            target_shape=resized.shape,
        )

        resized_affine = resized_geometry.affine

        # 10. Validate the final volume and its geometry.
        assert resized.shape == TARGET_SHAPE
        assert resized.dtype == np.float32
        assert np.isfinite(resized).all()
        assert resized.min() >= 0.0
        assert resized.max() <= 1.0

        assert resized_affine.shape == (4, 4)
        assert np.isfinite(resized_affine).all()
        assert np.allclose(
            resized_affine[3],
            [0.0, 0.0, 0.0, 1.0],
        )
        assert abs(
            np.linalg.det(resized_affine[:3, :3])
        ) > 1e-12

        # 11. Save the volume and its matching geometry.
        plane_dir = OUTPUT_ROOT / plane
        plane_dir.mkdir(parents=True, exist_ok=True)

        volume_file = plane_dir / "volume.npy"
        affine_file = plane_dir / "resized_affine.npy"

        np.save(volume_file, resized)
        np.save(affine_file, resized_affine)

        # Also retain the canonical affine for traceability.
        np.save(
            plane_dir / "canonical_affine.npy",
            grid.affine,
        )

        results.append({
            "plane": plane,
            "source_shape": source.volume.shape,
            "canonical_shape": normalized_volume.volume.shape,
            "final_shape": resized.shape,
            "volume_path": str(volume_file),
            "affine_path": str(affine_file),
        })

        print("Final shape:", resized.shape)
        print("Final intensity range:",
              float(resized.min()), "to", float(resized.max()))
        print("Resized affine: PASS")
        print("Saved volume:", volume_file)
        print("Saved affine:", affine_file)

if __name__ == "__main__":
    main()
