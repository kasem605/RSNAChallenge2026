from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "outputs" / "processed_dataset_v1"
MANIFEST_PATH = DATASET_DIR / "manifest.csv"

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "npz_zero_region_audit.csv"

PLANES = ("sagittal", "coronal", "axial")
SAMPLE_SIZE = 10


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest.dropna(
        subset=["study_instance_uid", "file_path"]
    ).head(SAMPLE_SIZE)

    results = []

    for _, row in manifest.iterrows():
        study_uid = str(row["study_instance_uid"])
        npz_path = (
            DATASET_DIR / str(row["file_path"])
        ).resolve()

        if not npz_path.is_relative_to(DATASET_DIR.resolve()):
            raise ValueError(f"Unexpected path: {npz_path}")

        if not npz_path.exists():
            print(f"Missing file: {npz_path}")
            continue

        with np.load(npz_path, allow_pickle=False) as data:
            for plane in PLANES:
                if plane not in data:
                    continue

                volume = np.asarray(data[plane])
                zero_mask = volume == 0

                # Zero fractions along each array axis.
                axis_fractions = [
                    zero_mask.mean(axis=tuple(
                        j for j in range(3) if j != axis
                    ))
                    for axis in range(3)
                ]

                # A boundary slice is considered mostly zero
                # when at least 95% of its voxels are zero.
                boundary_zero_slices = [
                    int(np.sum(fraction >= 0.95))
                    for fraction in axis_fractions
                ]

                results.append({
                    "study_uid": study_uid,
                    "plane": plane,
                    "shape": str(volume.shape),
                    "zero_fraction": float(zero_mask.mean()),
                    "axis0_zero_slice_min": float(axis_fractions[0].min()),
                    "axis0_zero_slice_max": float(axis_fractions[0].max()),
                    "axis1_zero_slice_min": float(axis_fractions[1].min()),
                    "axis1_zero_slice_max": float(axis_fractions[1].max()),
                    "axis2_zero_slice_min": float(axis_fractions[2].min()),
                    "axis2_zero_slice_max": float(axis_fractions[2].max()),
                    "mostly_zero_slices_axis0": boundary_zero_slices[0],
                    "mostly_zero_slices_axis1": boundary_zero_slices[1],
                    "mostly_zero_slices_axis2": boundary_zero_slices[2],
                })

    report = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)

    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(report.to_string(index=False))

    print(f"\nReport saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()