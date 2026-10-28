
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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

PLANES = ("sagittal", "coronal", "axial")
EXPECTED_SHAPE = (96, 160, 160)


class ProcessedKneeDataset(Dataset):
    """Load validated, preprocessed knee MRI volumes from disk."""

    def __init__(
        self,
        manifest_csv: str | Path,
        processed_csv: str | Path,
        study_uids: list[str] | None = None,
    ) -> None:
        self.manifest = pd.read_csv(manifest_csv)
        processed = pd.read_csv(processed_csv)

        required_manifest = {"StudyInstanceUID", *LABEL_COLUMNS}
        required_processed = {
            "StudyInstanceUID",
            "Plane",
            "VolumePath",
            "Status",
        }

        missing = required_manifest - set(self.manifest.columns)
        if missing:
            raise ValueError(
                f"Manifest is missing columns: {sorted(missing)}"
            )

        missing = required_processed - set(processed.columns)
        if missing:
            raise ValueError(
                f"Processed CSV is missing columns: {sorted(missing)}"
            )

        self.manifest["StudyInstanceUID"] = (
            self.manifest["StudyInstanceUID"].astype(str)
        )
        processed["StudyInstanceUID"] = (
            processed["StudyInstanceUID"].astype(str)
        )
        processed["Plane"] = processed["Plane"].astype(str).str.lower()

        # Use only studies with all 12 labels.
        self.manifest = self.manifest.dropna(
            subset=LABEL_COLUMNS
        ).copy()

        if study_uids is not None:
            requested = {str(uid) for uid in study_uids}
            self.manifest = self.manifest[
                self.manifest["StudyInstanceUID"].isin(requested)
            ].copy()

        if self.manifest["StudyInstanceUID"].duplicated().any():
            raise ValueError("Duplicate study IDs in manifest.")

        self.manifest = self.manifest.reset_index(drop=True)

        # Keep only successfully processed volumes.
        processed = processed[processed["Status"] == "SUCCESS"].copy()

        self.volume_paths: dict[tuple[str, str], Path] = {}

        for _, row in processed.iterrows():
            key = (row["StudyInstanceUID"], row["Plane"])

            if row["Plane"] not in PLANES:
                raise ValueError(f"Unexpected plane: {row['Plane']}")

            if key in self.volume_paths:
                raise ValueError(f"Duplicate processed record: {key}")

            self.volume_paths[key] = Path(row["VolumePath"])

        if self.manifest.empty:
            raise ValueError("No fully labeled studies found.")

        # Validate the three volume paths for every selected study.
        for uid in self.manifest["StudyInstanceUID"]:
            for plane in PLANES:
                key = (uid, plane)

                if key not in self.volume_paths:
                    raise ValueError(f"Missing processed volume: {key}")

                if not self.volume_paths[key].is_file():
                    raise FileNotFoundError(self.volume_paths[key])

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= len(self):
            raise IndexError(f"Dataset index out of range: {index}")

        row = self.manifest.iloc[index]
        study_uid = str(row["StudyInstanceUID"])

        volumes = []

        for plane in PLANES:
            path = self.volume_paths[(study_uid, plane)]
            volume = np.load(path, allow_pickle=False)

            if volume.shape != EXPECTED_SHAPE:
                raise ValueError(
                    f"{path}: expected {EXPECTED_SHAPE}, "
                    f"got {volume.shape}"
                )

            if not np.isfinite(volume).all():
                raise ValueError(f"Non-finite values in {path}")

            if volume.min() < 0 or volume.max() > 1:
                raise ValueError(f"Volume outside [0, 1]: {path}")

            volumes.append(
                torch.from_numpy(np.asarray(volume, dtype=np.float32))
            )

        images = torch.stack(volumes, dim=0)

        label_values = row[LABEL_COLUMNS].to_numpy(dtype=np.float32)

        if not np.isin(label_values, [0.0, 1.0]).all():
            raise ValueError(f"Non-binary labels for {study_uid}")

        return {
            "images": images,
            "labels": torch.from_numpy(label_values.copy()),
            "study_uid": study_uid,
        }
