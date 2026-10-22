
from pathlib import Path

import numpy as np
import pandas as pd

from ish_knee.dataset.knee_mri_dataset import KneeMRIDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "outputs" / "processed_dataset_v1"
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"

LABEL_NAMES = [
    "acl",
    "mcl",
    "medial_meniscus",
    "lateral_meniscus",
    "medial_oa",
    "lateral_oa",
    "pf_oa",
    "effusion",
    "synovitis",
    "bakers",
    "contusion",
    "fracture",
]

MANIFEST_LABEL_COLUMNS = [f"label_{name}" for name in LABEL_NAMES]


def resolve_path(value: str) -> Path:
    path = Path(value)

    if path.is_absolute() and path.exists():
        return path

    candidates = [
        PROJECT_ROOT / path,
        PROCESSED_DIR / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Cannot resolve file path: {value}")


def load_npz_labels(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "labels" not in data.files:
            raise KeyError(
                f"'labels' array missing from {path}. "
                f"Available arrays: {data.files}"
            )

        labels = np.asarray(data["labels"]).reshape(-1)

    if labels.shape != (12,):
        raise ValueError(
            f"Expected 12 labels in {path}, got {labels.shape}"
        )

    return labels.astype(np.float32)


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)

    missing_columns = [
        column
        for column in MANIFEST_LABEL_COLUMNS
        if column not in manifest.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing manifest label columns: {missing_columns}"
        )

    if manifest["study_instance_uid"].duplicated().any():
        raise ValueError("Duplicate study_instance_uid values found.")

    print("=" * 70)
    print("VERIFY LABEL ALIGNMENT")
    print("=" * 70)
    print(f"Manifest studies: {len(manifest)}")

    failures = []
    comparisons = 0

    for split in ["train", "validation"]:
        subset = manifest[
            manifest["split"].astype(str).str.lower() == split
        ].reset_index(drop=True)

        dataset = KneeMRIDataset(
            manifest_path=MANIFEST_PATH,
            split=split,
        )

        if len(dataset) != len(subset):
            raise ValueError(
                f"{split}: dataset has {len(dataset)} samples, "
                f"manifest has {len(subset)} rows"
            )

        print(f"\n{split.upper()}: {len(dataset)} studies")

        for index, row in subset.iterrows():
            study_uid = str(row["study_instance_uid"])

            try:
                npz_path = resolve_path(str(row["file_path"]))
                npz_labels = load_npz_labels(npz_path)

                manifest_labels = row[
                    MANIFEST_LABEL_COLUMNS
                ].to_numpy(dtype=np.float32)

                # Compare labels stored in the processed file
                # against the manifest's explicitly named columns.
                if not np.array_equal(npz_labels, manifest_labels):
                    differences = [
                        LABEL_NAMES[j]
                        for j in range(12)
                        if npz_labels[j] != manifest_labels[j]
                    ]

                    failures.append(
                        f"{study_uid}: NPZ vs manifest mismatch: "
                        f"{differences}"
                    )

                # Compare labels returned by the dataset.
                _, dataset_labels = dataset[index]
                dataset_labels = (
                    dataset_labels.detach().cpu().numpy().reshape(-1)
                )

                if dataset_labels.shape != (12,):
                    failures.append(
                        f"{study_uid}: dataset label shape "
                        f"{dataset_labels.shape}"
                    )
                elif not np.array_equal(
                    dataset_labels, manifest_labels
                ):
                    differences = [
                        LABEL_NAMES[j]
                        for j in range(12)
                        if dataset_labels[j] != manifest_labels[j]
                    ]

                    failures.append(
                        f"{study_uid}: dataset vs manifest mismatch: "
                        f"{differences}"
                    )

                comparisons += 1

            except Exception as exc:
                failures.append(f"{study_uid}: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Studies checked: {comparisons}")

    if failures:
        print(f"FAILURES: {len(failures)}")
        for failure in failures:
            print("FAIL:", failure)
        raise SystemExit(
            "Label alignment verification failed. "
            "Do not start another training run yet."
        )

    print("PASS: NPZ labels match manifest labels.")
    print("PASS: Dataset labels match manifest labels.")
    print("PASS: All checked labels use the expected 12-label order.")


if __name__ == "__main__":
    main()
