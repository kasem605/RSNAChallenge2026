from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "outputs" / "processed_dataset_v1"
MANIFEST_PATH = DATASET_DIR / "manifest.csv"

PLANES = ("sagittal", "coronal", "axial")


def inspect_volume(npz_path: Path, study_uid: str) -> list[dict]:
    results = []

    with np.load(npz_path, allow_pickle=False) as data:
        for plane in PLANES:
            if plane not in data:
                results.append({
                    "study_uid": study_uid,
                    "plane": plane,
                    "status": "MISSING_PLANE",
                })
                continue

            volume = data[plane]

            if volume.ndim != 3:
                results.append({
                    "study_uid": study_uid,
                    "plane": plane,
                    "shape": str(volume.shape),
                    "status": "INVALID_DIMENSIONS",
                })
                continue

            finite = np.isfinite(volume)
            finite_values = volume[finite]

            results.append({
                "study_uid": study_uid,
                "plane": plane,
                "shape": str(volume.shape),
                "dtype": str(volume.dtype),
                "finite_fraction": float(finite.mean()),
                "min": (
                    float(finite_values.min())
                    if finite_values.size else np.nan
                ),
                "max": (
                    float(finite_values.max())
                    if finite_values.size else np.nan
                ),
                "mean": (
                    float(finite_values.mean())
                    if finite_values.size else np.nan
                ),
                "std": (
                    float(finite_values.std())
                    if finite_values.size else np.nan
                ),
                "zero_fraction": float(np.mean(volume == 0)),
                "status": (
                    "OK"
                    if finite.all() and finite_values.size
                    else "NONFINITE_OR_EMPTY"
                ),
            })

    return results


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    manifest = pd.read_csv(MANIFEST_PATH)

    required = {"study_instance_uid", "file_path"}
    missing = required - set(manifest.columns)

    if missing:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing)}"
        )

    results = []

    # Audit a small sample first to keep the first run manageable.
    sample = manifest.dropna(subset=["study_instance_uid", "file_path"]).head(10)

    for _, row in sample.iterrows():
        relative_path = Path(str(row["file_path"]))
        npz_path = (DATASET_DIR / relative_path).resolve()

        if not npz_path.is_relative_to(DATASET_DIR.resolve()):
            raise ValueError(f"Unexpected path outside dataset: {npz_path}")

        if not npz_path.exists():
            print(f"MISSING FILE: {npz_path}")
            continue

        results.extend(
            inspect_volume(
                npz_path,
                str(row["study_instance_uid"]),
            )
        )

    report = pd.DataFrame(results)

    output_path = PROJECT_ROOT / "outputs" / "processed_npz_audit.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(report.to_string(index=False))

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()