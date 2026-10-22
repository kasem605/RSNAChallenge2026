
from pathlib import Path

import numpy as np

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.preprocessing.dicom.dicom_reader import (
    DicomSeriesReader,
)
from ish_knee.preprocessing.geometry.voxel_to_patient_affine import (
    calculate_voxel_to_patient_affine,
)


DATASET_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")


def main():
    print("=" * 70)
    print("VOXEL-TO-PATIENT AFFINE TEST")
    print("=" * 70)

    paths = DatasetPaths.from_root(DATASET_ROOT)
    metadata = MetadataReader(paths)
    metadata.validate()

    # Use the first study, matching your existing test workflow.
    study_uid = metadata.get_study_uids()[0]

    selector = SeriesSelector(
        train_series_dir=paths.train_series_dir
    )
    selection = selector.select(study_uid, metadata.train_series)

    reader = DicomSeriesReader()

    for plane in ("sagittal", "coronal", "axial"):
        print(f"\n{'-' * 70}")
        print(f"PLANE: {plane.upper()}")
        print("-" * 70)

        selected = getattr(selection, plane)

        if selected is None:
            print("No series selected; skipping.")
            continue

        series = reader.read(selected.series_path)
        ordered = series.ordered_slices()

        affine = calculate_voxel_to_patient_affine(series)

        positions = np.asarray(
            [s.image_position for s in ordered],
            dtype=np.float64,
        )

        indices = np.arange(len(ordered), dtype=np.float64)

        # Compare actual slice positions with the affine-predicted
        # position at row=0, column=0 for each slice.
        predicted_positions = (
            affine.origin[None, :]
            + indices[:, None] * affine.matrix[:, 0][None, :]
        )

        residuals = np.linalg.norm(
            positions - predicted_positions,
            axis=1,
        )

        # Round-trip test using a representative voxel.
        test_index = np.array([
            len(ordered) // 2,
            100.0,
            100.0,
        ])

        patient_position = affine.voxel_to_patient(
            tuple(test_index)
        )

        recovered_index = affine.patient_to_voxel(
            tuple(patient_position)
        )

        round_trip_error = np.max(
            np.abs(recovered_index - test_index)
        )

        print(f"Study UID: {series.study_instance_uid}")
        print(f"Series UID: {series.series_instance_uid}")
        print(f"Slice count: {len(ordered)}")
        print(f"Affine matrix:\n{affine.matrix}")
        print(f"Origin (LPS mm): {affine.origin}")
        print(f"Maximum position residual: {residuals.max():.6f} mm")
        print(f"Mean position residual: {residuals.mean():.6f} mm")
        print(f"Round-trip index error: {round_trip_error:.10f}")

        # These are diagnostic thresholds, not universal DICOM rules.
        if residuals.max() > 0.5:
            print("REVIEW: Slice positions are not sufficiently affine.")
        elif round_trip_error > 1e-6:
            print("FAIL: Voxel round-trip error is too large.")
        else:
            print("PASS: Affine geometry checks passed.")

    print("\n" + "=" * 70)
    print("AFFINE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
