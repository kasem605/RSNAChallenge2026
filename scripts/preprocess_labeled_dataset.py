
from pathlib import Path
import logging
import os

import numpy as np
import pandas as pd

from ish_knee.preprocessing.dicom.dicom_reader import (
    DicomSeriesReader,
)
from ish_knee.preprocessing.dicom.dicom_volume_builder import (
    DicomVolumeBuilder,
)
from ish_knee.preprocessing.dicom.dicom_volume import (
    DicomVolume,
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
from ish_knee.preprocessing.geometry.resize_geometry_transformer import (
    ResizeGeometryTransformer,
)
from ish_knee.preprocessing.volume.intensity_normalizer import (
    IntensityNormalizer,
)
from ish_knee.preprocessing.volume.volume_resizer import (
    VolumeResizer,
    VolumeResizeConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_manifest"
    / "study_series_manifest.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "processed_labeled_dataset"
)

TARGET_SHAPE = (96, 160, 160)
PLANES = ("sagittal", "coronal", "axial")

LABEL_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    """Write a NumPy array to a temporary file, then replace the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")

    try:
        with temporary_path.open("wb") as file:
            np.save(file, array)

        os.replace(temporary_path, path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def process_plane(
    study_uid: str,
    plane: str,
    series_uid: str,
    series_path: Path,
    reader: DicomSeriesReader,
    volume_builder: DicomVolumeBuilder,
    volume_resampler: VolumeResampler,
    intensity_normalizer: IntensityNormalizer,
    volume_resizer: VolumeResizer,
) -> dict:
    """Process one DICOM series and save its volume and geometry."""

    if not series_path.is_dir():
        raise FileNotFoundError(
            f"Series directory does not exist: {series_path}"
        )

    series = reader.read(series_path)

    # Verify that the manifest points to the expected study and series.
    if series.study_instance_uid != study_uid:
        raise ValueError(
            f"Study UID mismatch: expected {study_uid}, "
            f"received {series.study_instance_uid}"
        )

    if series.series_instance_uid != series_uid:
        raise ValueError(
            f"Series UID mismatch: expected {series_uid}, "
            f"received {series.series_instance_uid}"
        )

    # Build source volume from ordered DICOM slices.
    source = volume_builder.build(series)
    source_shape = source.volume.shape

    # Calculate source voxel-to-patient affine.
    affine_result = calculate_voxel_to_patient_affine(series)

    source_affine = np.eye(4, dtype=np.float64)
    source_affine[:3, :3] = affine_result.matrix
    source_affine[:3, 3] = affine_result.origin

    if not np.isfinite(source_affine).all():
        raise ValueError("Source affine contains non-finite values.")

    # Create a canonical physical grid at 1 mm spacing.
    grid = build_canonical_grid(
        source_shape=source_shape,
        source_affine=source_affine,
        spacing=1.0,
    )

    # Resample into canonical physical space.
    canonical_array = volume_resampler.resample(
        volume=source.volume,
        source_affine=source_affine,
        target_grid=grid,
    )

    if canonical_array.ndim != 3 or canonical_array.size == 0:
        raise ValueError(
            f"Invalid canonical volume shape: {canonical_array.shape}"
        )

    canonical_volume = DicomVolume(
        study_instance_uid=source.study_instance_uid,
        series_instance_uid=source.series_instance_uid,
        volume=canonical_array,
        source_path=source.source_path,
    )

    # Normalize intensity to [0, 1].
    normalized_volume = intensity_normalizer.normalize(
        canonical_volume
    )
    normalized = normalized_volume.volume

    # Resize to fixed model dimensions.
    resized = volume_resizer.resize(normalized)

    if resized.shape != TARGET_SHAPE:
        raise ValueError(
            f"Unexpected resized shape: {resized.shape}"
        )

    if resized.dtype != np.float32:
        raise ValueError(
            f"Unexpected dtype: {resized.dtype}"
        )

    if not np.isfinite(resized).all():
        raise ValueError("Resized volume contains non-finite values.")

    if float(resized.min()) < -1e-6 or float(resized.max()) > 1.000001:
        raise ValueError(
            "Normalized output is outside the expected [0, 1] range."
        )

    # Calculate geometry for the resized grid.
    resized_geometry = ResizeGeometryTransformer.transform(
        source_shape=normalized_volume.volume.shape,
        source_affine=grid.affine,
        target_shape=resized.shape,
    )

    resized_affine = resized_geometry.affine

    if resized_affine.shape != (4, 4):
        raise ValueError("Resized affine must have shape (4, 4).")

    if not np.isfinite(resized_affine).all():
        raise ValueError("Resized affine contains non-finite values.")

    if not np.allclose(
        resized_affine[3],
        [0.0, 0.0, 0.0, 1.0],
    ):
        raise ValueError("Invalid homogeneous affine bottom row.")

    if abs(np.linalg.det(resized_affine[:3, :3])) <= 1e-12:
        raise ValueError("Resized affine is singular.")

    # Store each study and plane separately.
    plane_dir = OUTPUT_ROOT / study_uid / plane
    plane_dir.mkdir(parents=True, exist_ok=True)

    volume_file = plane_dir / "volume.npy"
    canonical_affine_file = plane_dir / "canonical_affine.npy"
    resized_affine_file = plane_dir / "resized_affine.npy"

    atomic_save_npy(volume_file, resized)
    atomic_save_npy(canonical_affine_file, grid.affine)
    atomic_save_npy(resized_affine_file, resized_affine)

    # Verify the saved volume can be loaded and is valid.
    saved = np.load(volume_file, mmap_mode="r")

    if saved.shape != TARGET_SHAPE:
        raise ValueError(
            f"Saved volume has unexpected shape: {saved.shape}"
        )

    if saved.dtype != np.float32:
        raise ValueError(
            f"Saved volume has unexpected dtype: {saved.dtype}"
        )

    if not np.isfinite(saved).all():
        raise ValueError("Saved volume contains non-finite values.")

    return {
        "StudyInstanceUID": study_uid,
        "Plane": plane,
        "SeriesInstanceUID": series_uid,
        "SeriesPath": str(series_path),
        "SourceShape": str(source_shape),
        "CanonicalShape": str(canonical_volume.volume.shape),
        "ProcessedShape": str(TARGET_SHAPE),
        "VolumePath": str(volume_file),
        "CanonicalAffinePath": str(canonical_affine_file),
        "ResizedAffinePath": str(resized_affine_file),
        "VoxelMin": float(saved.min()),
        "VoxelMax": float(saved.max()),
        "VoxelDtype": str(saved.dtype),
        "Status": "SUCCESS",
    }


def main() -> None:
    print("=" * 70)
    print("LABELED DATASET BATCH PREPROCESSING")
    print("=" * 70)

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    required_columns = [
        "StudyInstanceUID",
        "AllPlanesAvailable",
        *LABEL_COLUMNS,
    ]

    for plane in PLANES:
        required_columns.extend([
            f"{plane}_SeriesInstanceUID",
            f"{plane}_Path",
        ])

    missing_columns = [
        col for col in required_columns
        if col not in manifest.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Manifest is missing columns: {missing_columns}"
        )

    if manifest["StudyInstanceUID"].isna().any():
        raise ValueError("Manifest contains missing study IDs.")

    if manifest["StudyInstanceUID"].duplicated().any():
        raise ValueError("Manifest contains duplicate study IDs.")

    if manifest[LABEL_COLUMNS].isna().any().any():
        raise ValueError(
            "Some labels are missing. "
            "Use the fully labeled manifest for this pipeline."
        )

    all_planes = (
        manifest["AllPlanesAvailable"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    if not all_planes.all():
        raise ValueError(
            "Manifest includes studies without all three planes."
        )

    # Initialize existing tested components.
    reader = DicomSeriesReader()
    volume_builder = DicomVolumeBuilder()

    volume_resampler = VolumeResampler(
        interpolation_order=1
    )

    intensity_normalizer = IntensityNormalizer()

    volume_resizer = VolumeResizer(
        VolumeResizeConfig(
            target_shape=TARGET_SHAPE,
            interpolation_order=1,
        )
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    successful_records = []
    failure_records = []

    total = len(manifest) * len(PLANES)
    completed = 0

    for _, row in manifest.iterrows():
        study_uid = str(row["StudyInstanceUID"])

        print(f"\nStudy: {study_uid}")

        for plane in PLANES:
            completed += 1

            series_uid = str(
                row[f"{plane}_SeriesInstanceUID"]
            )

            series_path = Path(
                str(row[f"{plane}_Path"])
            )

            print(
                f"[{completed}/{total}] "
                f"{study_uid} - {plane}"
            )

            try:
                record = process_plane(
                    study_uid=study_uid,
                    plane=plane,
                    series_uid=series_uid,
                    series_path=series_path,
                    reader=reader,
                    volume_builder=volume_builder,
                    volume_resampler=volume_resampler,
                    intensity_normalizer=intensity_normalizer,
                    volume_resizer=volume_resizer,
                )

                # Keep the original label values associated
                # with every processed plane.
                for label in LABEL_COLUMNS:
                    record[label] = row[label]

                successful_records.append(record)

                print(
                    f"  SUCCESS: {record['ProcessedShape']}"
                )

            except Exception as exc:
                logger.exception(
                    "Failed processing study %s, plane %s",
                    study_uid,
                    plane,
                )

                failure_records.append({
                    "StudyInstanceUID": study_uid,
                    "Plane": plane,
                    "SeriesInstanceUID": series_uid,
                    "SeriesPath": str(series_path),
                    "ErrorType": type(exc).__name__,
                    "ErrorMessage": str(exc),
                })

                print(f"  FAILED: {type(exc).__name__}: {exc}")

            # Persist progress after every plane so a long CPU
            # run still leaves a usable record if interrupted.
            pd.DataFrame(successful_records).to_csv(
                OUTPUT_ROOT / "processed_volumes.csv",
                index=False,
            )

            pd.DataFrame(failure_records).to_csv(
                OUTPUT_ROOT / "processing_failures.csv",
                index=False,
            )

    # Final summary.
    success_count = len(successful_records)
    failure_count = len(failure_records)

    successful_studies = {
        record["StudyInstanceUID"]
        for record in successful_records
    }

    studies_with_all_planes = sum(
        sum(
            record["StudyInstanceUID"] == study_uid
            for record in successful_records
        ) == len(PLANES)
        for study_uid in manifest["StudyInstanceUID"].astype(str)
    )

    print("\n" + "=" * 70)
    print("BATCH PREPROCESSING SUMMARY")
    print("=" * 70)
    print(f"Studies in manifest: {len(manifest)}")
    print(f"Expected volumes: {total}")
    print(f"Successful volumes: {success_count}")
    print(f"Failed volumes: {failure_count}")
    print(f"Studies with all three successful planes: {studies_with_all_planes}")
    print(f"Studies with at least one success: {len(successful_studies)}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print("=" * 70)

    if failure_count:
        print(
            "PREPROCESSING INCOMPLETE: "
            "Review outputs/processed_labeled_dataset/processing_failures.csv"
        )
        raise SystemExit(1)

    if success_count != total or studies_with_all_planes != len(manifest):
        print("PREPROCESSING INCOMPLETE: output count mismatch.")
        raise SystemExit(1)

    print("BATCH PREPROCESSING PASSED")


if __name__ == "__main__":
    main()
