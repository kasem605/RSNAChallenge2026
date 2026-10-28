
from datetime import datetime
from pathlib import Path
import csv
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder
from ish_knee.dataset.labeled_dataset_builder import LabeledDatasetBuilder
from ish_knee.model.study_split import StudySplit

from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
from ish_knee.preprocessing.volume.preprocessing_config import (
    PreprocessingConfig,
)
from ish_knee.preprocessing.volume.volume_preprocessor import (
    VolumePreprocessor,
)
from ish_knee.dataset.knee_mri_processor import KneeMRIProcessor

from ish_knee.model.model_sample_builder import ModelSampleBuilder
from ish_knee.model.model_dataset import ModelDataset
from ish_knee.model.pytorch_knee_dataset import PyTorchKneeDataset
from ish_knee.model.knee_3d_cnn import Knee3DCNN
from ish_knee.model.training_config import TrainingConfig
from ish_knee.model.knee_trainer import KneeTrainer

from ish_knee.model.label_weights import calculate_positive_weights


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATA_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET_SHAPE = (128, 128, 128)
TARGET_SPACING = VoxelSpacing(
    spacing_axis_0=1.0,
    spacing_axis_1=1.0,
    spacing_axis_2=1.0,
)

VALIDATION_FRACTION = 0.30
RANDOM_SEED = 42

BATCH_SIZE = 1
LEARNING_RATE = 0.001
EPOCHS = 10
NUM_WORKERS = 0

NUM_LABELS = 12


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_run_directory() -> Path:
    """Create a new run directory without overwriting prior runs."""
    base_dir = PROJECT_ROOT / "artifacts" / "training_runs"
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"knee_cnn_{timestamp}"

    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"knee_cnn_{timestamp}_{suffix}"
        suffix += 1

    run_dir.mkdir(parents=True)
    return run_dir


def validate_batch(
    sagittal: torch.Tensor,
    coronal: torch.Tensor,
    axial: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """Validate batch shapes, finite values and binary targets."""

    volumes = {
        "sagittal": sagittal,
        "coronal": coronal,
        "axial": axial,
    }

    for name, volume in volumes.items():
        if volume.ndim != 4:
            raise ValueError(
                f"{name} must have shape (batch, depth, height, width); "
                f"received {tuple(volume.shape)}"
            )

        if tuple(volume.shape[1:]) != TARGET_SHAPE:
            raise ValueError(
                f"{name} has unexpected volume shape "
                f"{tuple(volume.shape[1:])}"
            )

        if not torch.isfinite(volume).all():
            raise ValueError(f"{name} contains non-finite values.")

    if targets.ndim != 2 or targets.shape[1] != NUM_LABELS:
        raise ValueError(
            f"Targets must have shape (batch, {NUM_LABELS}); "
            f"received {tuple(targets.shape)}"
        )

    if not torch.isfinite(targets).all():
        raise ValueError("Targets contain non-finite values.")

    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("Targets must contain only binary 0/1 values.")

    batch_size = sagittal.shape[0]

    if not (
        coronal.shape[0] == batch_size
        and axial.shape[0] == batch_size
        and targets.shape[0] == batch_size
    ):
        raise ValueError("Batch sizes do not match.")


def move_batch_to_device(batch, device):
    sagittal, coronal, axial, targets = batch

    sagittal = sagittal.to(device, dtype=torch.float32)
    coronal = coronal.to(device, dtype=torch.float32)
    axial = axial.to(device, dtype=torch.float32)
    targets = targets.to(device, dtype=torch.float32)

    return sagittal, coronal, axial, targets


def validate_epoch(model, loader, criterion, device) -> float:
    """Evaluate without updating model parameters."""

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for batch in loader:
            sagittal, coronal, axial, targets = move_batch_to_device(
                batch, device
            )

            validate_batch(sagittal, coronal, axial, targets)

            logits = model(sagittal, coronal, axial)

            if logits.shape != targets.shape:
                raise ValueError(
                    f"Logits shape {tuple(logits.shape)} does not match "
                    f"target shape {tuple(targets.shape)}"
                )

            loss = criterion(logits, targets)

            if not torch.isfinite(loss):
                raise ValueError("Validation loss is not finite.")

            batch_size = targets.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Validation DataLoader contains no samples.")

    return total_loss / total_samples


def main() -> None:
    set_seed(RANDOM_SEED)

    device = torch.device("cpu")
    run_dir = create_run_directory()

    print("=" * 70)
    print("RSNA KNEE MODEL TRAINING")
    print("=" * 70)
    print(f"Device:       {device}")
    print(f"Run folder:   {run_dir}")
    print(f"Target shape: {TARGET_SHAPE}")
    print(f"Batch size:   {BATCH_SIZE}")
    print(f"Epochs:       {EPOCHS}")
    print("Class weights: enabled (per-label positive weights)")
    print()

    # --------------------------------------------------------
    # 1. Load metadata and build fully labeled studies
    # --------------------------------------------------------

    print("Loading dataset metadata...")

    paths = DatasetPaths.from_root(DATA_ROOT)
    metadata = MetadataReader(paths)
    metadata.validate()

    series_selector = SeriesSelector(paths.train_series_dir)
    sample_builder = DatasetSampleBuilder(metadata, series_selector)
    labeled_builder = LabeledDatasetBuilder(sample_builder)

    samples = labeled_builder.build_all()

    if len(samples) != 58:
        print(
            f"WARNING: expected 58 fully labeled studies; "
            f"found {len(samples)}."
        )

    if len(samples) < 2:
        raise ValueError("At least two fully labeled studies are required.")

    # --------------------------------------------------------
    # 2. Split studies before creating datasets
    # --------------------------------------------------------

    training_samples, validation_samples = StudySplit.split(
        samples,
        validation_fraction=VALIDATION_FRACTION,
        seed=RANDOM_SEED,
    )

    training_ids = {s.study_instance_uid for s in training_samples}
    validation_ids = {s.study_instance_uid for s in validation_samples}

    if training_ids & validation_ids:
        raise RuntimeError("Training and validation studies overlap.")

    print(f"Training studies:   {len(training_samples)}")
    print(f"Validation studies: {len(validation_samples)}")

    positive_weights = calculate_positive_weights(training_samples)

    print("\nPositive-class weights:")
    for label_name, weight in zip(
        [
            "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
            "Medial OA", "Lateral OA", "PF OA", "Effusion",
            "Synovitis", "Baker's", "Contusion", "Fracture",
        ],
        positive_weights,
    ):
        print(f"  {label_name:<20} {weight:.4f}")

    # Save the split IDs for reproducibility.
    with (run_dir / "study_split.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["StudyInstanceUID", "Split"])

        for study_id in sorted(training_ids):
            writer.writerow([study_id, "train"])

        for study_id in sorted(validation_ids):
            writer.writerow([study_id, "validation"])

    # --------------------------------------------------------
    # 3. Create the shared preprocessing pipeline
    # --------------------------------------------------------

    preprocessing_config = PreprocessingConfig(
        target_spacing=TARGET_SPACING,
        target_shape=TARGET_SHAPE,
    )

    volume_preprocessor = VolumePreprocessor(preprocessing_config)
    mri_processor = KneeMRIProcessor(volume_preprocessor)
    model_sample_builder = ModelSampleBuilder(mri_processor)

    # Separate datasets use separate study lists.
    training_model_dataset = ModelDataset(
        training_samples,
        model_sample_builder,
    )
    validation_model_dataset = ModelDataset(
        validation_samples,
        model_sample_builder,
    )

    training_dataset = PyTorchKneeDataset(training_model_dataset)
    validation_dataset = PyTorchKneeDataset(validation_model_dataset)

    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # --------------------------------------------------------
    # 4. Initialize model and trainer
    # --------------------------------------------------------

    config = TrainingConfig(
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        num_workers=NUM_WORKERS,
        checkpoint_dir=run_dir,
    )

    
    model = Knee3DCNN()

    trainer = KneeTrainer(
        model,
        config,
        device=device,
        positive_weights=positive_weights,
    )


    # Use the same loss function for training and validation.
    criterion = trainer.criterion

    best_validation_loss = float("inf")
    history_path = run_dir / "training_history.csv"

    # --------------------------------------------------------
    # 5. Train and validate
    # --------------------------------------------------------

    with history_path.open(
        "w", newline="", encoding="utf-8"
    ) as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=[
                "epoch",
                "training_loss",
                "validation_loss",
            ],
        )
        writer.writeheader()

        for epoch in range(1, config.epochs + 1):
            print()
            print("-" * 70)
            print(f"Epoch {epoch}/{config.epochs}")
            print("-" * 70)

            training_loss = trainer.train_epoch(training_loader)

            validation_loss = validate_epoch(
                trainer.model,
                validation_loader,
                criterion,
                trainer.device,
            )

            if not np.isfinite(training_loss):
                raise ValueError("Training loss is not finite.")

            print(f"Training loss:   {training_loss:.6f}")
            print(f"Validation loss: {validation_loss:.6f}")

            writer.writerow({
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
            })
            history_file.flush()

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "target_shape": TARGET_SHAPE,
                "target_spacing": (
                    TARGET_SPACING.spacing_axis_0,
                    TARGET_SPACING.spacing_axis_1,
                    TARGET_SPACING.spacing_axis_2,
                ),
                "seed": RANDOM_SEED,
                "positive_weights": positive_weights,
                "training_study_ids": sorted(training_ids),
                "validation_study_ids": sorted(validation_ids),
            }

            # Save the latest completed epoch.
            torch.save(checkpoint, run_dir / "latest.pt")

            # Preserve the best checkpoint from this run.
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(checkpoint, run_dir / "best.pt")
                print("New best validation checkpoint saved.")

    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Run folder: {run_dir}")
    print(f"History:    {history_path}")
    print(f"Best model: {run_dir / 'best.pt'}")
    print(f"Latest:     {run_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
