
"""
Train and evaluate the baseline 3D CNN for the RSNA Knee
Abnormality Detection AI Challenge.

Workflow:
1. Load and validate local dataset metadata.
2. Build studies with all 12 known binary labels.
3. Split studies into training and validation sets.
4. Preprocess sagittal, coronal, and axial MRI volumes.
5. Train Knee3DCNN.
6. Evaluate validation loss and per-label classification metrics.
7. Save the checkpoint, JSON results, and study split information.
"""
# Run from the project root:
#    python -u .\scripts\train_knee_baseline.py


import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


# ============================================================
# Project imports
# ============================================================

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader

from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder 
from ish_knee.dataset.labeled_dataset_builder import LabeledDatasetBuilder

from ish_knee.dataset.knee_mri_processor import KneeMRIProcessor

from ish_knee.preprocessing.volume.preprocessing_config import PreprocessingConfig
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
from ish_knee.preprocessing.volume.volume_preprocessor import VolumePreprocessor

from ish_knee.preprocessing.series_selector import SeriesSelector

from ish_knee.model.knee_3d_cnn import Knee3DCNN
from ish_knee.model.study_split import StudySplit
from ish_knee.model.model_dataset import ModelDataset
from ish_knee.model.model_sample_builder import ModelSampleBuilder
from ish_knee.model.pytorch_knee_dataset import PyTorchKneeDataset
from ish_knee.model.model_data_loader import ModelDataLoader
from ish_knee.model.knee_trainer import KneeTrainer
from ish_knee.model.training_config import TrainingConfig


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")

LABEL_NAMES = [
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

VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42

TARGET_SPACING = VoxelSpacing(
    spacing_axis_0=1.0,
    spacing_axis_1=1.0,
    spacing_axis_2=1.0,
)

TARGET_SHAPE = (64, 64, 64)

# Keep this at 1 for the initial baseline experiment.
EPOCHS = 1
BATCH_SIZE = 1
LEARNING_RATE = 0.001
NUM_WORKERS = 0

CHECKPOINT_DIR = Path("checkpoints")


# ============================================================
# Validation and evaluation
# ============================================================

def evaluate(model, loader, device):
    """Evaluate without updating model weights."""

    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    batch_count = 0
    all_logits = []
    all_targets = []

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

            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    if batch_count == 0:
        raise ValueError(
            "The validation DataLoader contains no batches."
        )

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0).to(torch.int32)

    # Convert model logits into binary predictions.
    predictions = (
        torch.sigmoid(logits) >= 0.5
    ).to(torch.int32)

    # Calculate confusion counts independently for each label.
    true_positive = (
        (predictions == 1) & (targets == 1)
    ).sum(dim=0)

    false_positive = (
        (predictions == 1) & (targets == 0)
    ).sum(dim=0)

    false_negative = (
        (predictions == 0) & (targets == 1)
    ).sum(dim=0)

    true_negative = (
        (predictions == 0) & (targets == 0)
    ).sum(dim=0)

    tp = true_positive.float()
    fp = false_positive.float()
    fn = false_negative.float()
    tn = true_negative.float()

    # Define precision and recall as zero when their
    # respective denominators are zero.
    precision = tp / (tp + fp).clamp(min=1.0)
    recall = tp / (tp + fn).clamp(min=1.0)

    f1 = (
        2.0 * precision * recall
    ) / (precision + recall).clamp(min=1e-8)

    accuracy = (
        (tp + tn) / (tp + tn + fp + fn).clamp(min=1.0)
    )

    per_label_metrics = {}

    for index, label in enumerate(LABEL_NAMES):
        per_label_metrics[label] = {
            "true_positive": int(tp[index].item()),
            "false_positive": int(fp[index].item()),
            "false_negative": int(fn[index].item()),
            "true_negative": int(tn[index].item()),
            "precision": round(
                precision[index].item(), 4
            ),
            "recall": round(
                recall[index].item(), 4
            ),
            "f1_score": round(
                f1[index].item(), 4
            ),
            "accuracy": round(
                accuracy[index].item(), 4
            ),
        }

    return {
        "validation_loss": total_loss / batch_count,
        "per_label_metrics": per_label_metrics,
    }


# ============================================================
# Utility functions
# ============================================================

def print_label_metrics(per_label_metrics):
    """Print validation metrics for each abnormality label."""

    print("\nPer-label validation metrics")
    print("=" * 100)

    print(
        f"{'Label':20s}"
        f"{'TP':>5s}"
        f"{'FP':>5s}"
        f"{'FN':>5s}"
        f"{'TN':>5s}"
        f"{'Precision':>12s}"
        f"{'Recall':>10s}"
        f"{'F1':>9s}"
        f"{'Accuracy':>11s}"
    )

    print("-" * 100)

    for label, metrics in per_label_metrics.items():
        print(
            f"{label:20s}"
            f"{metrics['true_positive']:5d}"
            f"{metrics['false_positive']:5d}"
            f"{metrics['false_negative']:5d}"
            f"{metrics['true_negative']:5d}"
            f"{metrics['precision']:12.3f}"
            f"{metrics['recall']:10.3f}"
            f"{metrics['f1_score']:9.3f}"
            f"{metrics['accuracy']:11.3f}"
        )

    print("=" * 100)


def save_json_results(results, output_path):
    """Write training and evaluation results to a JSON file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4,
            allow_nan=False,
        )

    print(f"\nTraining results saved to: {output_path.resolve()}")


# ============================================================
# Main training workflow
# ============================================================

def main():
    print("=" * 80)
    print("RSNA KNEE AI - BASELINE TRAINING")
    print("=" * 80)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Select the training device.
    # --------------------------------------------------------

    device = torch.device("cpu")

    print(f"\nDevice: {device}")
    print(f"Dataset root: {DATA_ROOT.resolve()}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR.resolve()}")

    # --------------------------------------------------------
    # 2. Load and validate metadata.
    # --------------------------------------------------------

    print("\n[1/8] Loading dataset metadata...")

    paths = DatasetPaths.from_root(DATA_ROOT)
    metadata = MetadataReader(paths)
    metadata.validate()

    print(f"Train studies: {len(metadata.train)}")
    print(f"Train series:  {len(metadata.train_series)}")

    # --------------------------------------------------------
    # 3. Build fully labeled studies.
    #
    # Missing labels must not be treated as negative labels.
    # The current labeled builder selects complete-label studies.
    # --------------------------------------------------------

    print("\n[2/8] Building fully labeled studies...")

    series_selector = SeriesSelector(paths.train_series_dir)

    dataset_sample_builder = DatasetSampleBuilder(
        metadata,
        series_selector,
    )

    labeled_dataset_builder = LabeledDatasetBuilder(
        dataset_sample_builder
    )

    samples = labeled_dataset_builder.build_all()

    if len(samples) < 2:
        raise ValueError(
            "At least two fully labeled studies are required "
            "for training and validation."
        )

    print(f"Fully labeled studies: {len(samples)}")

    # --------------------------------------------------------
    # 4. Split by study UID.
    #
    # All series belonging to a study remain in the same split.
    # --------------------------------------------------------

    print("\n[3/8] Splitting studies...")

    training_samples, validation_samples = StudySplit.split(
        samples,
        validation_fraction=VALIDATION_FRACTION,
        seed=RANDOM_SEED,
    )

    training_uids = [
        sample.study_instance_uid
        for sample in training_samples
    ]

    validation_uids = [
        sample.study_instance_uid
        for sample in validation_samples
    ]

    if set(training_uids) & set(validation_uids):
        raise RuntimeError(
            "Data leakage detected: training and validation "
            "study UIDs overlap."
        )

    print(f"Training studies:   {len(training_samples)}")
    print(f"Validation studies: {len(validation_samples)}")
    print("Study overlap check: PASSED")

    # --------------------------------------------------------
    # 5. Configure MRI preprocessing.
    # --------------------------------------------------------

    print("\n[4/8] Configuring MRI preprocessing...")

    preprocessing_config = PreprocessingConfig(
        target_spacing=TARGET_SPACING,
        target_shape=TARGET_SHAPE,
        normalize_intensity=True,
        padding_value=0.0,
    )

    volume_preprocessor = VolumePreprocessor(
        preprocessing_config
    )

    mri_processor = KneeMRIProcessor(
        volume_preprocessor
    )

    model_sample_builder = ModelSampleBuilder(
        mri_processor
    )

    # Create separate datasets from the study-level split.
    training_model_dataset = ModelDataset(
        training_samples,
        model_sample_builder,
    )

    validation_model_dataset = ModelDataset(
        validation_samples,
        model_sample_builder,
    )

    training_torch_dataset = PyTorchKneeDataset(
        training_model_dataset
    )

    validation_torch_dataset = PyTorchKneeDataset(
        validation_model_dataset
    )

    # --------------------------------------------------------
    # 6. Create DataLoaders.
    # --------------------------------------------------------

    print("\n[5/8] Creating DataLoaders...")

    training_loader = ModelDataLoader(
        training_torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    ).get_loader()

    validation_loader = ModelDataLoader(
        validation_torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    ).get_loader()

    print(f"Training batches:   {len(training_loader)}")
    print(f"Validation batches: {len(validation_loader)}")

    # --------------------------------------------------------
    # 7. Train the CNN baseline.
    # --------------------------------------------------------

    print("\n[6/8] Initializing and training the CNN...")

    model = Knee3DCNN()

    config = TrainingConfig(
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        num_workers=NUM_WORKERS,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    trainer = KneeTrainer(
        model=model,
        config=config,
        device=device,
    )

    training_losses = []

    for epoch in range(config.epochs):
        train_loss = trainer.train_epoch(training_loader)
        training_losses.append(float(train_loss))

        print(
            f"Epoch {epoch + 1}/{config.epochs} "
            f"- Training Loss: {train_loss:.6f}"
        )

    # --------------------------------------------------------
    # 8. Evaluate the model.
    # --------------------------------------------------------

    print("\n[7/8] Evaluating the validation set...")

    validation_results = evaluate(
        model,
        validation_loader,
        device,
    )

    val_loss = validation_results["validation_loss"]
    per_label_metrics = validation_results["per_label_metrics"]

    print(f"Validation Loss: {val_loss:.6f}")

    print_label_metrics(per_label_metrics)

    # --------------------------------------------------------
    # 9. Save the checkpoint.
    # --------------------------------------------------------

    print("\n[8/8] Saving model and results...")

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"knee_3d_cnn_baseline_epoch_{config.epochs}.pt"
    )

    torch.save(
        {
            "epoch": config.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "training_losses": training_losses,
            "validation_loss": val_loss,
            "label_names": LABEL_NAMES,
            "training_study_uids": training_uids,
            "validation_study_uids": validation_uids,
            "target_spacing": TARGET_SPACING.as_tuple,
            "target_shape": TARGET_SHAPE,
            "random_seed": RANDOM_SEED,
        },
        checkpoint_path,
    )

    print(f"Model checkpoint saved to: {checkpoint_path.resolve()}")

    # --------------------------------------------------------
    # 10. Save a JSON results report.
    # --------------------------------------------------------

    results = {
        "run_timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "device": str(device),
        "dataset_root": str(DATA_ROOT),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "random_seed": RANDOM_SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "target_spacing": list(TARGET_SPACING.as_tuple),
        "target_shape": list(TARGET_SHAPE),
        "total_fully_labeled_studies": len(samples),
        "training_study_count": len(training_samples),
        "validation_study_count": len(validation_samples),
        "training_losses": training_losses,
        "validation_loss": val_loss,
        "per_label_metrics": per_label_metrics,
        "label_names": LABEL_NAMES,
        "training_study_uids": training_uids,
        "validation_study_uids": validation_uids,
        "checkpoint_path": str(checkpoint_path.resolve()),
    }

    results_path = CHECKPOINT_DIR / "training_results.json"
    save_json_results(results, results_path)

    print("\n" + "=" * 80)
    print("BASELINE TRAINING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()

