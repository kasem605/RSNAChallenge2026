
import json
import random
from pathlib import Path

import numpy as np
import torch

from ish_knee.data import DatasetPaths, MetadataReader
from ish_knee.preprocessing import SeriesSelector
from ish_knee.preprocessing.volume.preprocessing_config import PreprocessingConfig
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
from ish_knee.preprocessing.volume.volume_preprocessor import VolumePreprocessor
from ish_knee.preprocessing.dicom.dicom_reader import DicomSeriesReader
from ish_knee.preprocessing.dicom.dicom_volume_builder import DicomVolumeBuilder
from ish_knee.preprocessing.volume.spacing_analyzer import SpacingAnalyzer

from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder
from ish_knee.dataset.labeled_dataset_builder import LabeledDatasetBuilder
from ish_knee.dataset.knee_mri_processor import KneeMRIProcessor

from ish_knee.model.model_sample_builder import ModelSampleBuilder
from ish_knee.model.model_dataset import ModelDataset
from ish_knee.model.pytorch_knee_dataset import PyTorchKneeDataset
from ish_knee.model.model_data_loader import ModelDataLoader
from ish_knee.model.knee_3d_cnn import Knee3DCNN
from ish_knee.model.knee_trainer import KneeTrainer
from ish_knee.model.training_config import TrainingConfig
from ish_knee.model.study_split import StudySplit


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

RANDOM_SEED = 42
SUBSET_SIZE = 4
EPOCHS = 100
BATCH_SIZE = 1
LEARNING_RATE = 0.001
NUM_WORKERS = 0

TARGET_SPACING = VoxelSpacing(1.0, 1.0, 1.0)
TARGET_SHAPE = (64, 64, 64)

CHECKPOINT_PATH = CHECKPOINT_DIR / "knee_3d_cnn_subset_100_epochs.pt"
RESULTS_PATH = CHECKPOINT_DIR / "subset_diagnostic_100_epoch_label_metrics.json"


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_loss(model, loader, device) -> float:
    """Calculate unweighted BCE loss without updating model weights."""
    criterion = torch.nn.BCEWithLogitsLoss()
    model.eval()

    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for sagittal, coronal, axial, targets in loader:
            sagittal = sagittal.to(device, dtype=torch.float32)
            coronal = coronal.to(device, dtype=torch.float32)
            axial = axial.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            logits = model(sagittal, coronal, axial)
            loss = criterion(logits, targets)

            total_loss += loss.item()
            batch_count += 1

    if batch_count == 0:
        raise ValueError("Evaluation DataLoader contains no batches.")

    return total_loss / batch_count


def main() -> None:
    set_random_seeds(RANDOM_SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SMALL-SUBSET CNN LEARNING DIAGNOSTIC")
    print("=" * 70)
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Subset size: {SUBSET_SIZE}")
    print(f"Epochs: {EPOCHS}")

    # --------------------------------------------------------
    # Load labeled studies
    # --------------------------------------------------------

    paths = DatasetPaths.from_root(DATASET_ROOT)
    metadata = MetadataReader(paths)
    metadata.validate()

    selector = SeriesSelector(paths.train_series_dir)
    sample_builder = DatasetSampleBuilder(metadata, selector)
    labeled_builder = LabeledDatasetBuilder(sample_builder)
    samples = labeled_builder.build_all()

    print(f"Fully labeled studies: {len(samples)}")

    # Use the same study-level split as the original experiments.
    training_samples, _ = StudySplit.split(
        samples,
        validation_fraction=0.30,
        seed=RANDOM_SEED,
    )

    if len(training_samples) < SUBSET_SIZE:
        raise ValueError("Not enough training studies for the diagnostic.")

    subset_samples = training_samples[:SUBSET_SIZE]

    print("\nDiagnostic study UIDs:")
    for sample in subset_samples:
        print(sample.study_instance_uid)

    # --------------------------------------------------------
    # Build the existing preprocessing and dataset pipeline
    # --------------------------------------------------------

    preprocessing_config = PreprocessingConfig(
        target_spacing=TARGET_SPACING,
        target_shape=TARGET_SHAPE,
        normalize_intensity=True,
        padding_value=0.0,
    )

    preprocessor = VolumePreprocessor(preprocessing_config)
    mri_processor = KneeMRIProcessor(preprocessor)
    model_sample_builder = ModelSampleBuilder(mri_processor)

    model_dataset = ModelDataset(subset_samples, model_sample_builder)
    torch_dataset = PyTorchKneeDataset(model_dataset)

    loader = ModelDataLoader(
        torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    ).get_loader()

    evaluation_loader = ModelDataLoader(
        torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    ).get_loader()

    # --------------------------------------------------------
    # Train the diagnostic model from scratch
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Continue training from the existing 20-epoch checkpoint
    # --------------------------------------------------------

    SOURCE_CHECKPOINT = (
        CHECKPOINT_DIR / "knee_3d_cnn_subset_100_epochs.pt"
    )

    model = Knee3DCNN()

    checkpoint = torch.load(
        SOURCE_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded model weights from: {SOURCE_CHECKPOINT}")

    config = TrainingConfig(
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        num_workers=NUM_WORKERS,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    trainer = KneeTrainer(
        model=model,
        config=config,
        device=device,
    )

    training_losses = []
    evaluation_losses = []

    for epoch in range(20, EPOCHS):
        train_loss = trainer.train_epoch(loader)
        eval_loss = evaluate_loss(model, evaluation_loader, device)

        training_losses.append(float(train_loss))
        evaluation_losses.append(float(eval_loss))

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.6f} | "
            f"Same-subset eval loss: {eval_loss:.6f}"
        )

    # --------------------------------------------------------
    # Save separate diagnostic artifacts
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "study_uids": [
                sample.study_instance_uid for sample in subset_samples
            ],
            "target_shape": TARGET_SHAPE,
            "random_seed": RANDOM_SEED,
            "epochs": EPOCHS,
        },
        CHECKPOINT_PATH,
    )

    results = {
        "diagnostic": "small_subset_overfit",
        "study_count": len(subset_samples),
        "study_uids": [
            sample.study_instance_uid for sample in subset_samples
        ],
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "target_spacing": list(TARGET_SPACING.as_tuple),
        "target_shape": list(TARGET_SHAPE),
        "training_losses": training_losses,
        "same_subset_evaluation_losses": evaluation_losses,
        "checkpoint_path": str(CHECKPOINT_PATH),
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nDiagnostic complete.")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Results:    {RESULTS_PATH}")
    print(f"Initial training loss: {training_losses[0]:.6f}")
    print(f"Final training loss:   {training_losses[-1]:.6f}")
    print(f"Final same-subset evaluation loss: {evaluation_losses[-1]:.6f}")


if __name__ == "__main__":
    main()
