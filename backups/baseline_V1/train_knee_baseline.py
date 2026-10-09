
import json
from datetime import datetime, timezone

from pathlib import Path

import torch
from torch import nn

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader

from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder
from ish_knee.dataset.labeled_dataset_builder import LabeledDatasetBuilder
from ish_knee.dataset.knee_mri_processor import KneeMRIProcessor

from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.preprocessing.volume.preprocessing_config import PreprocessingConfig
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
from ish_knee.preprocessing.volume.volume_preprocessor import VolumePreprocessor

from ish_knee.model.study_split import StudySplit
from ish_knee.model.model_dataset import ModelDataset
from ish_knee.model.model_sample_builder import ModelSampleBuilder
from ish_knee.model.pytorch_knee_dataset import PyTorchKneeDataset
from ish_knee.model.model_data_loader import ModelDataLoader
from ish_knee.model.knee_3d_cnn import Knee3DCNN
from ish_knee.model.knee_trainer import KneeTrainer
from ish_knee.model.training_config import TrainingConfig


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
        raise ValueError("The validation DataLoader contains no batches.")

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    predictions = (torch.sigmoid(logits) >= 0.5).to(torch.int32)
    correct = (predictions == targets.to(torch.int32)).float()

    per_label_accuracy = correct.mean(dim=0)

    return total_loss / batch_count, per_label_accuracy


def main():
    print("=" * 70)
    print("RSNA KNEE CNN BASELINE TRAINING")
    print("=" * 70)

    # Reproducibility
    torch.manual_seed(42)

    # --------------------------------------------------
    # 1. Load metadata and fully labeled studies
    # --------------------------------------------------
    dataset_root = Path(r"D:\RSNA knee abnormality detection 2026")
    paths = DatasetPaths.from_root(dataset_root)

    metadata = MetadataReader(paths)
    metadata.validate()

    series_selector = SeriesSelector(paths.train_series_dir)

    sample_builder = DatasetSampleBuilder(
        metadata=metadata,
        series_selector=series_selector,
    )

    labeled_builder = LabeledDatasetBuilder(sample_builder)
    samples = labeled_builder.build_all()

    print("Fully labeled studies:", len(samples))

    if len(samples) < 2:
        raise ValueError("At least two labeled studies are required.")

    # --------------------------------------------------
    # 2. Create the reproducible study-level split
    # --------------------------------------------------
    training_samples, validation_samples = StudySplit.split(
        samples,
        validation_fraction=0.20,
        seed=42,
    )

    training_uids = {
        sample.study_instance_uid for sample in training_samples
    }
    validation_uids = {
        sample.study_instance_uid for sample in validation_samples
    }

    assert training_uids.isdisjoint(validation_uids)

    print("Training studies:", len(training_samples))
    print("Validation studies:", len(validation_samples))
    print("Study overlap check: PASSED")

    # --------------------------------------------------
    # 3. Configure MRI preprocessing
    # --------------------------------------------------
    preprocessing_config = PreprocessingConfig(
        target_spacing=VoxelSpacing(1.0, 1.0, 1.0),
        target_shape=(64, 64, 64),
    )

    preprocessor = VolumePreprocessor(preprocessing_config)
    mri_processor = KneeMRIProcessor(preprocessor)
    model_sample_builder = ModelSampleBuilder(mri_processor)

    # --------------------------------------------------
    # 4. Create independent datasets and DataLoaders
    # --------------------------------------------------
    training_model_dataset = ModelDataset(
        samples=training_samples,
        sample_builder=model_sample_builder,
    )

    validation_model_dataset = ModelDataset(
        samples=validation_samples,
        sample_builder=model_sample_builder,
    )

    training_torch_dataset = PyTorchKneeDataset(training_model_dataset)
    validation_torch_dataset = PyTorchKneeDataset(validation_model_dataset)

    config = TrainingConfig(
        batch_size=1,
        learning_rate=0.001,
        epochs=1,
        num_workers=0,
        checkpoint_dir=Path("checkpoints"),
    )

    training_loader = ModelDataLoader(
        dataset=training_torch_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    ).get_loader()

    validation_loader = ModelDataLoader(
        dataset=validation_torch_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    ).get_loader()

    # --------------------------------------------------
    # 5. Initialize and train the CNN on CPU
    # --------------------------------------------------
    device = torch.device("cpu")
    model = Knee3DCNN()

    trainer = KneeTrainer(
        model=model,
        config=config,
        device=device,
    )

    print()
    print("Device:", device)
    print("Starting training...")
    print("MRI preprocessing is performed as studies are loaded.")
    print("This first run may take some time.")

    for epoch in range(config.epochs):
        training_loss = trainer.train_epoch(training_loader)

        print(
            f"Epoch {epoch + 1}/{config.epochs} "
            f"- Training loss: {training_loss:.6f}"
        )

    # --------------------------------------------------
    # 6. Evaluate on held-out studies
    # --------------------------------------------------
    validation_loss, per_label_accuracy = evaluate(
        model,
        validation_loader,
        device,
    )

    print()
    print(f"Validation loss: {validation_loss:.6f}")
    print("Per-label validation accuracy (12 studies):")

    for name, accuracy in zip(LABEL_NAMES, per_label_accuracy):
        print(f"  {name:20s}: {accuracy.item():.3f}")

    
    # --------------------------------------------------
    # Save structured training results
    # --------------------------------------------------
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "fully_labeled_studies": len(samples),
        "training_studies": len(training_samples),
        "validation_studies": len(validation_samples),
        "training_loss": float(training_loss),
        "validation_loss": float(validation_loss),
        "per_label_validation_accuracy": {
            name: float(accuracy.item())
            for name, accuracy in zip(LABEL_NAMES, per_label_accuracy)
        },
        "training_study_uids": sorted(training_uids),
        "validation_study_uids": sorted(validation_uids),
    }

    results_path = config.checkpoint_dir / "training_results.json"

    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("Training results saved to:", results_path.resolve())
    
    # --------------------------------------------------
    # 7. Save the model checkpoint
    # --------------------------------------------------
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        config.checkpoint_dir / "knee_3d_cnn_baseline_epoch_1.pt"
    )

    torch.save(
        {
            "epoch": config.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "label_names": LABEL_NAMES,
            "training_study_uids": sorted(training_uids),
            "validation_study_uids": sorted(validation_uids),
        },
        checkpoint_path,
    )

    print()
    print("Checkpoint saved to:", checkpoint_path.resolve())
    print("BASELINE TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()