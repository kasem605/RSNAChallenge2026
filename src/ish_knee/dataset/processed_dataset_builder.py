
from pathlib import Path
import csv

import numpy as np

from ..dataset.dataset_sample import DatasetSample
from ..model.study_split import StudySplit
from .knee_mri_processor import KneeMRIProcessor


LABEL_NAMES = (
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
)


class ProcessedDatasetBuilder:
    """Preprocess labeled studies and save them individually to disk."""

    def __init__(
        self,
        processor: KneeMRIProcessor,
        output_dir: Path,
        target_shape: tuple[int, int, int] = (128, 128, 128),
    ) -> None:
        if not isinstance(processor, KneeMRIProcessor):
            raise TypeError("processor must be a KneeMRIProcessor instance")

        if len(target_shape) != 3 or any(size <= 0 for size in target_shape):
            raise ValueError("target_shape must contain three positive dimensions")

        self._processor = processor
        self._output_dir = Path(output_dir)
        self._target_shape = target_shape

    def build(
        self,
        samples: list[DatasetSample],
        validation_fraction: float = 0.30,
        seed: int = 42,
    ) -> Path:
        """Split studies, process them, and write a manifest."""

        if not samples:
            raise ValueError("No labeled samples were supplied")

        study_uids = [sample.study_instance_uid for sample in samples]

        if len(study_uids) != len(set(study_uids)):
            raise ValueError("Duplicate study UIDs found")

        for sample in samples:
            if not sample.study.is_complete:
                raise ValueError(
                    f"Study {sample.study_instance_uid} is missing an MRI plane"
                )

        # Refuse to overwrite an earlier dataset.
        if self._output_dir.exists():
            raise FileExistsError(
                f"Output directory already exists: {self._output_dir}. "
                "Choose a new output directory to preserve existing results."
            )

        train_samples, validation_samples = StudySplit.split(
            samples,
            validation_fraction=validation_fraction,
            seed=seed,
        )

        self._output_dir.mkdir(parents=True)
        (self._output_dir / "train").mkdir()
        (self._output_dir / "validation").mkdir()

        manifest_rows = []

        splits = (
            ("train", train_samples),
            ("validation", validation_samples),
        )

        for split_name, split_samples in splits:
            for index, sample in enumerate(split_samples, start=1):
                uid = sample.study_instance_uid
                print(
                    f"[{split_name}] {index}/{len(split_samples)} "
                    f"Processing study {uid}"
                )

                mri = self._processor.process(sample)

                volumes = {
                    "sagittal": mri.sagittal,
                    "coronal": mri.coronal,
                    "axial": mri.axial,
                }

                for plane, volume in volumes.items():
                    if volume.shape != self._target_shape:
                        raise ValueError(
                            f"{uid} {plane}: expected shape "
                            f"{self._target_shape}, got {volume.shape}"
                        )

                    if volume.dtype != np.float32:
                        raise TypeError(
                            f"{uid} {plane}: expected float32, got {volume.dtype}"
                        )

                    if not np.isfinite(volume).all():
                        raise ValueError(
                            f"{uid} {plane} contains NaN or infinite values"
                        )

                labels = np.asarray(sample.labels.as_tuple, dtype=np.float32)

                if labels.shape != (len(LABEL_NAMES),):
                    raise ValueError(
                        f"{uid}: expected {len(LABEL_NAMES)} labels, "
                        f"got {labels.shape}"
                    )

                if not np.isin(labels, (0.0, 1.0)).all():
                    raise ValueError(f"{uid}: labels must contain only 0 or 1")

                relative_path = Path(split_name) / f"{uid}.npz"
                file_path = self._output_dir / relative_path

                # One study per file; no dataset-wide volume accumulation.
                np.savez_compressed(
                    file_path,
                    sagittal=mri.sagittal,
                    coronal=mri.coronal,
                    axial=mri.axial,
                    labels=labels,
                )

                row = {
                    "study_instance_uid": uid,
                    "split": split_name,
                    "file_path": relative_path.as_posix(),
                }
                row.update(
                    {
                        f"label_{name}": int(value)
                        for name, value in zip(LABEL_NAMES, labels)
                    }
                )
                manifest_rows.append(row)

        manifest_path = self._output_dir / "manifest.csv"

        fieldnames = [
            "study_instance_uid",
            "split",
            "file_path",
            *(f"label_{name}" for name in LABEL_NAMES),
        ]

        with manifest_path.open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

        self._validate_manifest(
            self._output_dir,
            manifest_rows,
            train_samples,
            validation_samples,
        )

        print("\nProcessed dataset created successfully.")
        print(f"Training studies:   {len(train_samples)}")
        print(f"Validation studies: {len(validation_samples)}")
        print(f"Manifest: {manifest_path}")
        print(f"Output directory: {self._output_dir}")

        return self._output_dir

    
    @staticmethod
    def _validate_manifest(
        output_dir: Path,
        rows: list[dict],
        train_samples: list[DatasetSample],
        validation_samples: list[DatasetSample],
    ) -> None:
        train_uids = {s.study_instance_uid for s in train_samples}
        validation_uids = {s.study_instance_uid for s in validation_samples}
        saved_uids = {row["study_instance_uid"] for row in rows}

        if train_uids & validation_uids:
            raise ValueError("Training and validation studies overlap")

        if saved_uids != train_uids | validation_uids:
            raise ValueError("Manifest study UIDs do not match the split")

        if len(rows) != len(saved_uids):
            raise ValueError("Manifest contains duplicate study UIDs")

        for row in rows:
            file_path = output_dir / row["file_path"]

            if not file_path.is_file():
                raise FileNotFoundError(
                    f"Manifest references a missing file: {file_path}"
                )
