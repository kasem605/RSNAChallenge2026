"""
Evaluate thresholds for the saved weighted full-training model.
Run from project root:
"""
#  python -u .\scripts\evaluate_weighted_thresholds.py
"""

Uses validation study UIDs stored in the checkpoint. Does not retrain or
modify existing checkpoints/results.
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
THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60)
TARGET_SPACING = VoxelSpacing(1.0, 1.0, 1.0)
TARGET_SHAPE = (64, 64, 64)
BATCH_SIZE = 1
NUM_WORKERS = 0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / CHECKPOINT_NAME
OUTPUT_DIR = CHECKPOINT_DIR / "threshold_evaluation"


def calculate_metrics(targets, probabilities, threshold, label_names):
    predictions = probabilities >= threshold
    rows, f1_values = [], []
    for i, label in enumerate(label_names):
        actual = targets[:, i].astype(bool)
        predicted = predictions[:, i]
        tp = int(np.logical_and(actual, predicted).sum())
        fp = int(np.logical_and(~actual, predicted).sum())
        fn = int(np.logical_and(actual, ~predicted).sum())
        tn = int(np.logical_and(~actual, ~predicted).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
        rows.append({
            "threshold": threshold, "label": label, "tp": tp, "fp": fp,
            "fn": fn, "tn": tn, "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
        })
        f1_values.append(f1)
    return rows, float(np.mean(f1_values)) if f1_values else 0.0


def main():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    print("=" * 76)
    print("RSNA KNEE AI - WEIGHTED MODEL THRESHOLD EVALUATION")
    print("=" * 76)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    validation_uids = checkpoint.get("validation_study_uids")
    label_names = checkpoint.get("label_names")
    if not validation_uids or not label_names:
        raise ValueError("Checkpoint must contain validation_study_uids and label_names.")

    print("\n[1/5] Rebuilding labeled studies and restoring saved validation split...")
    paths = DatasetPaths.from_root(DATA_ROOT)
    metadata = MetadataReader(paths)
    metadata.validate()
    selector = SeriesSelector(paths.train_series_dir)
    sample_builder = DatasetSampleBuilder(metadata, selector)
    all_samples = LabeledDatasetBuilder(sample_builder).build_all()
    samples_by_uid = {sample.study_instance_uid: sample for sample in all_samples}
    missing = [uid for uid in validation_uids if uid not in samples_by_uid]
    if missing:
        raise ValueError(f"{len(missing)} saved validation UIDs missing; first: {missing[0]}")
    validation_samples = [samples_by_uid[uid] for uid in validation_uids]
    print(f"Validation studies restored: {len(validation_samples)}")

    print("\n[2/5] Applying the same MRI preprocessing...")
    config = PreprocessingConfig(
        target_spacing=TARGET_SPACING, target_shape=TARGET_SHAPE,
        normalize_intensity=True, padding_value=0.0,
    )
    processor = KneeMRIProcessor(VolumePreprocessor(config))
    model_sample_builder = ModelSampleBuilder(processor)
    model_dataset = ModelDataset(validation_samples, model_sample_builder)
    torch_dataset = PyTorchKneeDataset(model_dataset)
    loader = ModelDataLoader(
        torch_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    ).get_loader()
    print(f"Validation batches: {len(loader)}")

    print("\n[3/5] Loading model and calculating probabilities...")
    model = Knee3DCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    probability_batches, target_batches = [], []
    with torch.no_grad():
        for sagittal, coronal, axial, targets in loader:
            logits = model(
                sagittal.to(dtype=torch.float32),
                coronal.to(dtype=torch.float32),
                axial.to(dtype=torch.float32),
            )
            probability_batches.append(torch.sigmoid(logits).cpu().numpy())
            target_batches.append(targets.cpu().numpy())
    probabilities = np.concatenate(probability_batches, axis=0)
    targets = np.concatenate(target_batches, axis=0).astype(np.int32)
    if probabilities.shape != targets.shape or probabilities.shape[1] != len(label_names):
        raise ValueError(f"Unexpected output/target shapes: {probabilities.shape}, {targets.shape}")
    print(f"Probability matrix: {probabilities.shape}")

    print("\n[4/5] Evaluating thresholds...")
    all_rows, summary = [], []
    for threshold in (0.20, 0.30, 0.40, 0.50, 0.60):
        rows, macro_f1 = calculate_metrics(targets, probabilities, threshold, label_names)
        all_rows.extend(rows)
        summary.append({"threshold": threshold, "macro_f1": round(macro_f1, 4)})
        print(f"Threshold {threshold:.2f} | Macro F1: {macro_f1:.4f}")
    best = max(summary, key=lambda item: item["macro_f1"])

    print("\n[5/5] Saving new reports (existing files are untouched)...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "weighted_threshold_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    probabilities_path = OUTPUT_DIR / "weighted_validation_probabilities.csv"
    with probabilities_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["study_instance_uid"]
        fields += [f"{name}_actual" for name in label_names]
        fields += [f"{name}_probability" for name in label_names]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row_index, sample in enumerate(validation_samples):
            row = {"study_instance_uid": sample.study_instance_uid}
            for i, name in enumerate(label_names):
                row[f"{name}_actual"] = int(targets[row_index, i])
                row[f"{name}_probability"] = round(float(probabilities[row_index, i]), 6)
            writer.writerow(row)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(CHECKPOINT_PATH.resolve()),
        "validation_study_count": len(validation_samples),
        "validation_study_uids": list(validation_uids),
        "label_names": list(label_names),
        "macro_f1_by_threshold": summary,
        "highest_macro_f1_on_validation": best,
        "note": "Threshold selection on this validation set is diagnostic, not an unbiased final test estimate.",
    }
    json_path = OUTPUT_DIR / "weighted_threshold_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    print("\nMacro F1 summary:")
    for item in summary:
        suffix = "  <-- highest on this validation set" if item == best else ""
        print(f"  {item['threshold']:.2f}: {item['macro_f1']:.4f}{suffix}")
    print("\nSaved:")
    print(metrics_path.resolve())
    print(probabilities_path.resolve())
    print(json_path.resolve())
    print("\nCompleted. No model checkpoint was modified.")


if __name__ == "__main__":
    main()
