"""
Compare the saved baseline and weighted RSNA Knee CNN checkpoints.
No model training occurs in this script.
Run from the project root:
"""
#    python -u .\scripts\evaluate_knee_checkpoints.py


import csv
import json
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

from ish_knee.model.knee_3d_cnn import Knee3DCNN
from ish_knee.model.model_dataset import ModelDataset
from ish_knee.model.model_sample_builder import ModelSampleBuilder
from ish_knee.model.pytorch_knee_dataset import PyTorchKneeDataset
from ish_knee.model.model_data_loader import ModelDataLoader


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATA_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = CHECKPOINT_DIR / "evaluation"

CHECKPOINTS = {
    "baseline": CHECKPOINT_DIR / "knee_3d_cnn_baseline_epoch_1.pt",
    "weighted": CHECKPOINT_DIR / "knee_3d_cnn_weighted_epoch_1.pt",
}

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

TARGET_SPACING = VoxelSpacing(1.0, 1.0, 1.0)
TARGET_SHAPE = (64, 64, 64)
BATCH_SIZE = 1
NUM_WORKERS = 0
THRESHOLD = 0.5


# ------------------------------------------------------------
# Checkpoint loading and validation
# ------------------------------------------------------------

def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path.resolve()}")

    # Only load checkpoints produced by this trusted local project.
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = {
        "model_state_dict",
        "validation_study_uids",
        "label_names",
        "target_spacing",
        "target_shape",
    }

    missing = required_keys - checkpoint.keys()
    if missing:
        raise ValueError(
            f"{path.name} is missing checkpoint fields: {sorted(missing)}"
        )

    if list(checkpoint["label_names"]) != LABEL_NAMES:
        raise ValueError(
            f"Label order in {path.name} does not match this evaluator."
        )

    if tuple(checkpoint["target_shape"]) != TARGET_SHAPE:
        raise ValueError(
            f"Unexpected target shape in {path.name}: "
            f"{checkpoint['target_shape']}"
        )

    saved_spacing = tuple(checkpoint["target_spacing"])
    if saved_spacing != TARGET_SPACING.as_tuple:
        raise ValueError(
            f"Unexpected target spacing in {path.name}: {saved_spacing}"
        )

    if not checkpoint["validation_study_uids"]:
        raise ValueError(f"No validation study IDs in {path.name}")

    return checkpoint


# ------------------------------------------------------------
# Dataset construction
# ------------------------------------------------------------

def build_validation_dataset(validation_uids, metadata, paths):
    """Rebuild only the validation studies saved in the checkpoints."""

    train = metadata.train
    uid_column = "StudyInstanceUID"

    if train[uid_column].duplicated().any():
        raise ValueError(
            "train.csv contains duplicate StudyInstanceUID values."
        )

    train_by_uid = train.set_index(uid_column)
    missing_uids = [
        uid for uid in validation_uids
        if uid not in train_by_uid.index
    ]

    if missing_uids:
        raise ValueError(
            f"{len(missing_uids)} validation studies are missing "
            f"from train.csv. First missing UID: {missing_uids[0]}"
        )

    incomplete = []
    for uid in validation_uids:
        row = train_by_uid.loc[uid]
        missing_labels = [
            label for label in LABEL_NAMES
            if label not in row.index or not _is_present(row[label])
        ]
        if missing_labels:
            incomplete.append((uid, missing_labels))

    if incomplete:
        raise ValueError(
            "Validation studies contain missing labels. "
            f"First example: {incomplete[0]}"
        )

    selector = SeriesSelector(paths.train_series_dir)
    sample_builder = DatasetSampleBuilder(metadata, selector)

    # Preserve the exact order saved in the checkpoint.
    samples = [
        sample_builder.build(uid)
        for uid in validation_uids
    ]

    for sample in samples:
        if not sample.study.is_complete:
            raise ValueError(
                f"Study {sample.study_instance_uid} does not have "
                "all three required MRI planes."
            )

    preprocessing_config = PreprocessingConfig(
        target_spacing=TARGET_SPACING,
        target_shape=TARGET_SHAPE,
        normalize_intensity=True,
        padding_value=0.0,
    )

    volume_preprocessor = VolumePreprocessor(preprocessing_config)
    mri_processor = KneeMRIProcessor(volume_preprocessor)
    model_sample_builder = ModelSampleBuilder(mri_processor)

    model_dataset = ModelDataset(samples, model_sample_builder)
    torch_dataset = PyTorchKneeDataset(model_dataset)

    loader = ModelDataLoader(
        torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    ).get_loader()

    return loader


def _is_present(value):
    """Check whether a pandas scalar is missing."""
    import pandas as pd
    return not pd.isna(value)


# ------------------------------------------------------------
# Model inference
# ------------------------------------------------------------

def predict_checkpoint(name, checkpoint, loader, validation_uids):
    """Return validation probabilities, targets, and average batch loss."""

    model = Knee3DCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch.device("cpu"))
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    probabilities = []
    targets_all = []
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for sagittal, coronal, axial, targets in loader:
            sagittal = sagittal.float()
            coronal = coronal.float()
            axial = axial.float()
            targets = targets.float()

            logits = model(sagittal, coronal, axial)
            loss = criterion(logits, targets)

            total_loss += loss.item()
            batch_count += 1

            probabilities.append(torch.sigmoid(logits).cpu())
            targets_all.append(targets.cpu())

    if batch_count != len(validation_uids):
        raise RuntimeError(
            f"{name}: expected {len(validation_uids)} validation batches, "
            f"received {batch_count}."
        )

    probs = torch.cat(probabilities, dim=0).numpy()
    targets = torch.cat(targets_all, dim=0).numpy()

    if probs.shape != (len(validation_uids), len(LABEL_NAMES)):
        raise ValueError(f"{name}: unexpected prediction shape {probs.shape}")

    return {
        "probabilities": probs,
        "targets": targets.astype(int),
        "validation_loss": total_loss / batch_count,
    }


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def calculate_metrics(targets, probabilities):
    predictions = (probabilities >= THRESHOLD).astype(int)
    metrics = {}

    for index, label in enumerate(LABEL_NAMES):
        actual = targets[:, index]
        predicted = predictions[:, index]

        tp = int(((actual == 1) & (predicted == 1)).sum())
        fp = int(((actual == 0) & (predicted == 1)).sum())
        fn = int(((actual == 1) & (predicted == 0)).sum())
        tn = int(((actual == 0) & (predicted == 0)).sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)

        metrics[label] = {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(accuracy, 4),
        }

    return metrics


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("=" * 76)
    print("RSNA KNEE AI - CHECKPOINT EVALUATION")
    print("=" * 76)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Loading checkpoints...")
    checkpoints = {
        name: load_checkpoint(path)
        for name, path in CHECKPOINTS.items()
    }

    baseline_uids = list(checkpoints["baseline"]["validation_study_uids"])
    weighted_uids = list(checkpoints["weighted"]["validation_study_uids"])

    if baseline_uids != weighted_uids:
        raise ValueError(
            "Baseline and weighted checkpoints do not have the same "
            "validation study IDs in the same order."
        )

    validation_uids = baseline_uids
    print(f"Validation studies: {len(validation_uids)}")
    print("Validation UID comparison: PASSED")

    print("\n[2/5] Loading and validating metadata...")
    paths = DatasetPaths.from_root(DATA_ROOT)
    metadata = MetadataReader(paths)
    metadata.validate()

    print("\n[3/5] Building the exact validation DataLoader...")
    loader = build_validation_dataset(validation_uids, metadata, paths)
    print(f"Validation batches: {len(loader)}")

    print("\n[4/5] Evaluating saved checkpoints...")
    results = {}
    for name in ("baseline", "weighted"):
        print(f"Evaluating {name}...")
        results[name] = predict_checkpoint(
            name,
            checkpoints[name],
            loader,
            validation_uids,
        )
        print(
            f"{name} validation loss: "
            f"{results[name]['validation_loss']:.6f}"
        )

    baseline_targets = results["baseline"]["targets"]
    weighted_targets = results["weighted"]["targets"]

    if not (baseline_targets == weighted_targets).all():
        raise RuntimeError(
            "The two models received different validation targets."
        )

    for name in ("baseline", "weighted"):
        results[name]["per_label_metrics"] = calculate_metrics(
            results[name]["targets"],
            results[name]["probabilities"],
        )

    print("\n[5/5] Saving evaluation reports...")

    # One row per study; actual labels and both models' probabilities.
    prediction_path = OUTPUT_DIR / "study_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "StudyInstanceUID",
                *[
                    column
                    for label in LABEL_NAMES
                    for column in (
                        f"actual__{label}",
                        f"baseline_probability__{label}",
                        f"baseline_prediction__{label}",
                        f"weighted_probability__{label}",
                        f"weighted_prediction__{label}",
                    )
                ],
            ],
        )
        writer.writeheader()

        baseline_probs = results["baseline"]["probabilities"]
        weighted_probs = results["weighted"]["probabilities"]
        targets = baseline_targets

        for study_index, uid in enumerate(validation_uids):
            row = {"StudyInstanceUID": uid}

            for label_index, label in enumerate(LABEL_NAMES):
                baseline_probability = float(
                    baseline_probs[study_index, label_index]
                )
                weighted_probability = float(
                    weighted_probs[study_index, label_index]
                )

                row[f"actual__{label}"] = int(
                    targets[study_index, label_index]
                )
                row[f"baseline_probability__{label}"] = round(
                    baseline_probability, 6
                )
                row[f"baseline_prediction__{label}"] = int(
                    baseline_probability >= THRESHOLD
                )
                row[f"weighted_probability__{label}"] = round(
                    weighted_probability, 6
                )
                row[f"weighted_prediction__{label}"] = int(
                    weighted_probability >= THRESHOLD
                )

            writer.writerow(row)

    # Side-by-side metrics for each label.
    comparison_path = OUTPUT_DIR / "label_metrics_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "label",
            "baseline_f1",
            "weighted_f1",
            "f1_difference",
            "baseline_precision",
            "weighted_precision",
            "baseline_recall",
            "weighted_recall",
            "baseline_tp",
            "weighted_tp",
            "baseline_fp",
            "weighted_fp",
            "baseline_fn",
            "weighted_fn",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for label in LABEL_NAMES:
            base = results["baseline"]["per_label_metrics"][label]
            weighted = results["weighted"]["per_label_metrics"][label]

            writer.writerow({
                "label": label,
                "baseline_f1": base["f1_score"],
                "weighted_f1": weighted["f1_score"],
                "f1_difference": round(
                    weighted["f1_score"] - base["f1_score"], 4
                ),
                "baseline_precision": base["precision"],
                "weighted_precision": weighted["precision"],
                "baseline_recall": base["recall"],
                "weighted_recall": weighted["recall"],
                "baseline_tp": base["true_positive"],
                "weighted_tp": weighted["true_positive"],
                "baseline_fp": base["false_positive"],
                "weighted_fp": weighted["false_positive"],
                "baseline_fn": base["false_negative"],
                "weighted_fn": weighted["false_negative"],
            })

    # Detailed JSON report, including the split and all per-label metrics.
    json_path = OUTPUT_DIR / "checkpoint_comparison.json"
    report = {
        "validation_study_count": len(validation_uids),
        "validation_study_uids": validation_uids,
        "threshold": THRESHOLD,
        "target_spacing": list(TARGET_SPACING.as_tuple),
        "target_shape": list(TARGET_SHAPE),
        "models": {
            name: {
                "checkpoint": str(CHECKPOINTS[name]),
                "checkpoint_epoch": checkpoints[name].get("epoch"),
                "saved_validation_loss": checkpoints[name].get(
                    "validation_loss"
                ),
                "recomputed_validation_loss": results[name]["validation_loss"],
                "per_label_metrics": results[name]["per_label_metrics"],
            }
            for name in ("baseline", "weighted")
        },
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, allow_nan=False)

    print("\nEvaluation complete.")
    print(f"Study predictions: {prediction_path.resolve()}")
    print(f"Metric comparison: {comparison_path.resolve()}")
    print(f"Detailed JSON:     {json_path.resolve()}")

    print("\nPer-label F1 comparison")
    print("-" * 65)
    print(f"{'Label':20s}{'Baseline':>12s}{'Weighted':>12s}{'Difference':>14s}")

    for label in LABEL_NAMES:
        base_f1 = results["baseline"]["per_label_metrics"][label]["f1_score"]
        weighted_f1 = results["weighted"]["per_label_metrics"][label]["f1_score"]
        print(
            f"{label:20s}{base_f1:12.4f}{weighted_f1:12.4f}"
            f"{weighted_f1 - base_f1:14.4f}"
        )


if __name__ == "__main__":
    main()
