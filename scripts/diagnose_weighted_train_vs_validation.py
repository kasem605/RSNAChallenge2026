"""
Compare the saved weighted model's predictions on training vs validation studies.

Run from the project root:
"""
#    python -u .\scripts\diagnose_weighted_train_vs_validation.py
"""
This script does not retrain or modify any checkpoint. It restores the exact
validation studies recorded in the checkpoint and treats all remaining labeled
studies as the training set.
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

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
from ish_knee.model.model_dataset import ModelDataset
from ish_knee.model.model_sample_builder import ModelSampleBuilder
from ish_knee.model.pytorch_knee_dataset import PyTorchKneeDataset
from ish_knee.model.model_data_loader import ModelDataLoader

DATA_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")
CHECKPOINT_NAME = "knee_3d_cnn_full_training_weighted_epoch_20.pt"
TARGET_SPACING = VoxelSpacing(1.0, 1.0, 1.0)
TARGET_SHAPE = (64, 64, 64)
BATCH_SIZE = 1
NUM_WORKERS = 0
THRESHOLDS = (0.40, 0.50)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / CHECKPOINT_NAME
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "train_validation_diagnostic"


def metrics_for_threshold(targets, probabilities, threshold, label_names, split_name):
    predicted = probabilities >= threshold
    rows = []
    f1_values = []
    for i, label in enumerate(label_names):
        actual = targets[:, i].astype(bool)
        pred = predicted[:, i]
        tp = int(np.logical_and(actual, pred).sum())
        fp = int(np.logical_and(~actual, pred).sum())
        fn = int(np.logical_and(actual, ~pred).sum())
        tn = int(np.logical_and(~actual, ~pred).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
        rows.append({
            "split": split_name,
            "threshold": threshold,
            "label": label,
            "positive_actual": int(actual.sum()),
            "positive_predicted": int(pred.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
        })
        f1_values.append(f1)
    return rows, float(np.mean(f1_values)) if f1_values else 0.0


def infer_split(samples, processor, model, label_names, split_name):
    model_sample_builder = ModelSampleBuilder(processor)
    model_dataset = ModelDataset(samples, model_sample_builder)
    torch_dataset = PyTorchKneeDataset(model_dataset)
    loader = ModelDataLoader(
        torch_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    ).get_loader()

    probability_batches, target_batches = [], []
    with torch.no_grad():
        for batch_index, (sagittal, coronal, axial, targets) in enumerate(loader, start=1):
            logits = model(
                sagittal.to(dtype=torch.float32),
                coronal.to(dtype=torch.float32),
                axial.to(dtype=torch.float32),
            )
            probability_batches.append(torch.sigmoid(logits).cpu().numpy())
            target_batches.append(targets.cpu().numpy())
            if batch_index % 5 == 0 or batch_index == len(loader):
                print(f"  {split_name}: processed {batch_index}/{len(loader)} studies")

    probabilities = np.concatenate(probability_batches, axis=0)
    targets = np.concatenate(target_batches, axis=0).astype(np.int32)
    if probabilities.shape != targets.shape or probabilities.shape[1] != len(label_names):
        raise ValueError(
            f"{split_name}: unexpected output/target shapes "
            f"{probabilities.shape} vs {targets.shape}"
        )
    return targets, probabilities


def main():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    print("=" * 78)
    print("RSNA KNEE AI - TRAINING VS VALIDATION DIAGNOSTIC")
    print("=" * 78)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    validation_uids = checkpoint.get("validation_study_uids")
    label_names = checkpoint.get("label_names")
    if not validation_uids or not label_names:
        raise ValueError("Checkpoint must contain validation_study_uids and label_names.")

    print("\n[1/5] Rebuilding labeled studies...")
    paths = DatasetPaths.from_root(DATA_ROOT)
    metadata = MetadataReader(paths)
    metadata.validate()
    selector = SeriesSelector(paths.train_series_dir)
    sample_builder = DatasetSampleBuilder(metadata, selector)
    all_samples = LabeledDatasetBuilder(sample_builder).build_all()
    samples_by_uid = {str(s.study_instance_uid): s for s in all_samples}

    validation_uids = [str(uid) for uid in validation_uids]
    missing = [uid for uid in validation_uids if uid not in samples_by_uid]
    if missing:
        raise ValueError(f"{len(missing)} checkpoint validation UIDs missing; first: {missing[0]}")

    validation_samples = [samples_by_uid[uid] for uid in validation_uids]
    validation_uid_set = set(validation_uids)
    training_samples = [s for s in all_samples if str(s.study_instance_uid) not in validation_uid_set]
    if not training_samples or not validation_samples:
        raise ValueError("Training or validation split is empty.")

    print(f"Total labeled studies: {len(all_samples)}")
    print(f"Training studies (remaining studies): {len(training_samples)}")
    print(f"Validation studies (restored from checkpoint): {len(validation_samples)}")
    print("Note: training set is inferred as all labeled studies not in the saved validation UID list.")

    print("\n[2/5] Configuring the same preprocessing...")
    config = PreprocessingConfig(
        target_spacing=TARGET_SPACING,
        target_shape=TARGET_SHAPE,
        normalize_intensity=True,
        padding_value=0.0,
    )
    processor = KneeMRIProcessor(VolumePreprocessor(config))

    print("\n[3/5] Loading the saved model...")
    model = Knee3DCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\n[4/5] Evaluating training studies...")
    train_targets, train_probs = infer_split(
        training_samples, processor, model, label_names, "Training"
    )
    print("\n[5/5] Evaluating validation studies...")
    val_targets, val_probs = infer_split(
        validation_samples, processor, model, label_names, "Validation"
    )

    all_metric_rows = []
    summary = []
    split_results = {
        "training": (training_samples, train_targets, train_probs),
        "validation": (validation_samples, val_targets, val_probs),
    }

    print("\nMacro F1 comparison:")
    for split_name, (_, targets, probabilities) in split_results.items():
        for threshold in THRESHOLDS:
            rows, macro_f1 = metrics_for_threshold(
                targets, probabilities, threshold, label_names, split_name
            )
            all_metric_rows.extend(rows)
            summary.append({
                "split": split_name,
                "study_count": int(len(targets)),
                "threshold": threshold,
                "macro_f1": round(macro_f1, 4),
                "mean_probability": round(float(probabilities.mean()), 6),
                "std_probability": round(float(probabilities.std()), 6),
                "fraction_predictions_positive": round(float((probabilities >= threshold).mean()), 6),
            })
            print(
                f"  {split_name:10s} threshold {threshold:.2f}: "
                f"macro F1={macro_f1:.4f}; "
                f"mean probability={probabilities.mean():.4f}; "
                f"std={probabilities.std():.4f}; "
                f"positive predictions={(probabilities >= threshold).mean():.1%}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "train_validation_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_metric_rows)

    probability_path = OUTPUT_DIR / "train_validation_probabilities.csv"
    with probability_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["split", "study_instance_uid"]
        fields += [f"{name}_actual" for name in label_names]
        fields += [f"{name}_probability" for name in label_names]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split_name, (samples, targets, probabilities) in split_results.items():
            for row_index, sample in enumerate(samples):
                row = {
                    "split": split_name,
                    "study_instance_uid": str(sample.study_instance_uid),
                }
                for i, name in enumerate(label_names):
                    row[f"{name}_actual"] = int(targets[row_index, i])
                    row[f"{name}_probability"] = round(float(probabilities[row_index, i]), 6)
                writer.writerow(row)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(CHECKPOINT_PATH.resolve()),
        "total_labeled_studies": len(all_samples),
        "training_study_count": len(training_samples),
        "validation_study_count": len(validation_samples),
        "validation_study_uids": validation_uids,
        "label_names": list(label_names),
        "results": summary,
        "interpretation_note": (
            "Training metrics are in-sample diagnostics; validation metrics estimate "
            "generalization more directly but are noisy because the validation set is small. "
            "Training studies are all labeled studies not listed as validation in the checkpoint."
        ),
    }
    json_path = OUTPUT_DIR / "train_validation_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    print("\nSaved new diagnostic files (no existing files overwritten):")
    print(metrics_path.resolve())
    print(probability_path.resolve())
    print(json_path.resolve())
    print("\nCompleted. No model checkpoint was modified.")


if __name__ == "__main__":
    main()
