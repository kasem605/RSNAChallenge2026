
from pathlib import Path
import csv

import numpy as np
import torch
from torch.utils.data import Dataset


class KneeMRIDataset(Dataset):
    """
    PyTorch dataset for preprocessed knee MRI studies.

    Each item contains:
        images: float32 tensor, shape (3, 128, 128, 128)
        labels: float32 tensor, shape (12,)
    """

    PLANES = ("sagittal", "coronal", "axial")
    EXPECTED_SHAPE = (128, 128, 128)
    LABEL_COUNT = 12

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()

        if split not in ("train", "validation"):
            raise ValueError(
                "split must be 'train' or 'validation'"
            )

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        self.dataset_dir = self.manifest_path.parent

        with self.manifest_path.open(
            "r", newline="", encoding="utf-8"
        ) as file:
            rows = list(csv.DictReader(file))

        self.samples = [
            row for row in rows
            if row["split"] == split
        ]

        if not self.samples:
            raise ValueError(
                f"No samples found for split '{split}'"
            )

        self.split = split

        print(
            f"KneeMRIDataset initialized: "
            f"{self.split} = {len(self.samples)} studies"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        row = self.samples[index]
        study_uid = row["study_instance_uid"]

        file_path = (
            self.dataset_dir / row["file_path"]
        ).resolve()

        # Ensure the manifest cannot reference files outside
        # this processed dataset directory.
        if not file_path.is_relative_to(self.dataset_dir):
            raise ValueError(
                f"Invalid file path for study {study_uid}"
            )

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Processed study file not found: {file_path}"
            )

        with np.load(file_path, allow_pickle=False) as data:
            volumes = []

            for plane in self.PLANES:
                volume = data[plane]

                if volume.shape != self.EXPECTED_SHAPE:
                    raise ValueError(
                        f"{study_uid} {plane}: "
                        f"unexpected shape {volume.shape}"
                    )

                if volume.dtype != np.float32:
                    raise TypeError(
                        f"{study_uid} {plane}: "
                        f"expected float32, got {volume.dtype}"
                    )

                if not np.isfinite(volume).all():
                    raise ValueError(
                        f"{study_uid} {plane}: "
                        "contains NaN or infinity"
                    )

                volumes.append(volume)

            images_array = np.stack(volumes, axis=0)

            labels_array = data["labels"].copy()

        if labels_array.shape != (self.LABEL_COUNT,):
            raise ValueError(
                f"{study_uid}: unexpected label shape "
                f"{labels_array.shape}"
            )

        if not np.isin(labels_array, (0, 1)).all():
            raise ValueError(
                f"{study_uid}: labels must be binary"
            )

        # Copy to ensure the tensor owns writable memory.
        images = torch.from_numpy(
            np.ascontiguousarray(images_array)
        )
        labels = torch.from_numpy(
            np.asarray(labels_array, dtype=np.float32).copy()
        )

        return images, labels
