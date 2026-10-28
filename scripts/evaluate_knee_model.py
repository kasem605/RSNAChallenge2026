
from datetime import datetime
from pathlib import Path
import csv
import math

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


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATA_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Use the run that produced your reported best validation loss.
RUN_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "training_runs"
    / "knee_cnn_20261019_171331"
)

CHECKPOINT_PATH = RUN_DIR / "best.pt"

TARGET_SHAPE = (128, 128, 128)

TARGET_SPACING = VoxelSpacing(
    spacing_axis_0=1.0,
    spacing_axis_1=1.0,
    spacing_axis_2=1.0,
)

RANDOM_SEED = 42
BATCH_SIZE = 1
NUM_WORKERS = 0
THRESHOLD = 0.5

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


# ------------------------------------------------------------
# Output folders
# ------------------------------------------------------------

def create_evaluation_directory() -> Path:
    """Create a unique directory; preserve all previous results."""

    base_dir = PROJECT_ROOT / "artifacts" / "evaluations"
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"knee_eval_{timestamp}"

    suffix = 1
    while output_dir.exists():
        output_dir = base_dir / f"knee_eval_{timestamp}_{suffix}"
        suffix += 1

    output_dir.mkdir(parents=True)
    return output_dir


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def calculate_auc(
    actual: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """
    Calculate ROC-AUC when both classes are present.

    Returns NaN if AUC cannot be calculated.
    """

    if len(np.unique(actual)) < 2:
        return float("nan")

    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(actual, probabilities))
    except ImportError:
        return float("nan")


def calculate_metrics(
    actual: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    """Calculate binary classification metrics for one label."""

    predicted = (probabilities >= threshold).astype(np.int32)
    actual = actual.astype(np.int32)

    tp = int(np.sum((actual == 1) & (predicted == 1)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))

    total = len(actual)

    accuracy = (tp + tn) / total if total else float("nan")
    sensitivity = (
        tp / (tp + fn) if (tp + fn) else float("nan")
    )
    specificity = (
        tn / (tn + fp) if (tn + fp) else float("nan")
    )
    precision = (
        tp / (tp + fp) if (tp + fp) else float("nan")
    )

    if math.isfinite(precision) and math.isfinite(sensitivity):
        denominator = precision + sensitivity
        f1 = (
            2 * precision * sensitivity / denominator
            if denominator > 0
            else 0.0
        )
    else:
        f1 = float("nan")

    return {
        "positive_count": int(np.sum(actual == 1)),
        "negative_count": int(np.sum(actual == 0)),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "roc_auc": calculate_auc(actual, probabilities),
    }


# ------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------

def main() -> None:
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cpu")

    print("=" * 70)
    print("RSNA KNEE MODEL VALIDATION EVALUATION")
    print("=" * 70)

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}\n"
            "Check RUN_DIR against your actual training run."
        )

    # Load the checkpoint on CPU.
    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
        )

    required_keys = {
        "model_state_dict",
        "validation_study_ids",
        "target_shape",
        "target_spacing",
    }

    missing_keys = required_keys - set(checkpoint.keys())
    if missing_keys:
        raise KeyError(
            f"Checkpoint is missing required fields: {missing_keys}"
        )

    if tuple(checkpoint["target_shape"]) != TARGET_SHAPE:
        raise ValueError(
            "Checkpoint target shape does not match this script."
        )

    if len(LABEL_NAMES) != 12:
        raise ValueError("Exactly 12 label names are required.")

    saved_validation_ids = checkpoint["validation_study_ids"]

    if len(saved_validation_ids) != len(set(saved_validation_ids)):
        raise ValueError(
            "Checkpoint contains duplicate validation study IDs."
        )

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Saved validation loss: {checkpoint['validation_loss']:.6f}")
    print(f"Device: {device}")
    print()

    # --------------------------------------------------------
    # 1. Load metadata and rebuild fully labeled samples
    # --------------------------------------------------------

    print("Loading metadata and labeled studies...")

    paths = DatasetPaths.from_root(DATA_ROOT)
    metadata = MetadataReader(paths)
    metadata.validate()

    series_selector = SeriesSelector(paths.train_series_dir)
    sample_builder = DatasetSampleBuilder(metadata, series_selector)
    labeled_builder = LabeledDatasetBuilder(sample_builder)

    all_samples = labeled_builder.build_all()

    samples_by_id = {
        sample.study_instance_uid: sample
        for sample in all_samples
    }

    if len(samples_by_id) != len(all_samples):
        raise ValueError(
            "Duplicate study IDs were found in the labeled samples."
        )

    missing_ids = [
        study_id
        for study_id in saved_validation_ids
        if study_id not in samples_by_id
    ]

    if missing_ids:
        raise ValueError(
            "Some saved validation studies could not be rebuilt. "
            f"Missing count: {len(missing_ids)}. "
            f"First missing IDs: {missing_ids[:3]}"
        )

    # Preserve the exact validation ID order stored in the checkpoint.
    validation_samples = [
        samples_by_id[study_id]
        for study_id in saved_validation_ids
    ]

    if len(validation_samples) != len(saved_validation_ids):
        raise RuntimeError("Validation sample count does not match.")

    # Confirm that validation IDs do not overlap training IDs.
    training_ids = set(checkpoint.get("training_study_ids", []))
    validation_ids = set(saved_validation_ids)

    if training_ids & validation_ids:
        raise ValueError(
            "Training and validation study IDs overlap."
        )

    print(f"Rebuilt labeled studies: {len(all_samples)}")
    print(f"Validation studies: {len(validation_samples)}")

    # --------------------------------------------------------
    # 2. Rebuild the same preprocessing pipeline
    # --------------------------------------------------------

    preprocessing_config = PreprocessingConfig(
        target_spacing=TARGET_SPACING,
        target_shape=TARGET_SHAPE,
    )

    volume_preprocessor = VolumePreprocessor(preprocessing_config)
    mri_processor = KneeMRIProcessor(volume_preprocessor)
    model_sample_builder = ModelSampleBuilder(mri_processor)

    validation_model_dataset = ModelDataset(
        validation_samples,
        model_sample_builder,
    )

    validation_dataset = PyTorchKneeDataset(
        validation_model_dataset
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # --------------------------------------------------------
    # 3. Load trained model weights
    # --------------------------------------------------------

    model = Knee3DCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --------------------------------------------------------
    # 4. Run inference
    # --------------------------------------------------------

    actual_batches = []
    probability_batches = []

    print()
    print("Running validation inference...")

    with torch.inference_mode():
        for batch_index, batch in enumerate(validation_loader):
            if len(batch) != 4:
                raise ValueError(
                    "Expected sagittal, coronal, axial and targets."
                )

            sagittal, coronal, axial, targets = batch

            sagittal = sagittal.to(device, dtype=torch.float32)
            coronal = coronal.to(device, dtype=torch.float32)
            axial = axial.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            for name, volume in (
                ("sagittal", sagittal),
                ("coronal", coronal),
                ("axial", axial),
            ):
                if not torch.isfinite(volume).all():
                    raise ValueError(
                        f"{name} volume contains non-finite values."
                    )

                if tuple(volume.shape[1:]) != TARGET_SHAPE:
                    raise ValueError(
                        f"Unexpected {name} shape: "
                        f"{tuple(volume.shape)}"
                    )

            if targets.shape != (sagittal.shape[0], 12):
                raise ValueError(
                    f"Unexpected target shape: {tuple(targets.shape)}"
                )

            if not torch.isfinite(targets).all():
                raise ValueError("Targets contain non-finite values.")

            if not torch.all((targets == 0) | (targets == 1)):
                raise ValueError(
                    "Validation targets must be binary."
                )

            logits = model(sagittal, coronal, axial)

            if logits.shape != targets.shape:
                raise ValueError(
                    f"Logits shape {tuple(logits.shape)} "
                    f"does not match targets {tuple(targets.shape)}."
                )

            probabilities = torch.sigmoid(logits)

            if not torch.isfinite(probabilities).all():
                raise ValueError(
                    "Model produced non-finite probabilities."
                )

            actual_batches.append(targets.cpu().numpy())
            probability_batches.append(probabilities.cpu().numpy())

            if (batch_index + 1) % 5 == 0:
                print(
                    f"Processed {batch_index + 1}/"
                    f"{len(validation_dataset)} studies"
                )

    if not actual_batches:
        raise ValueError("No validation predictions were generated.")

    actual_matrix = np.concatenate(actual_batches, axis=0)
    probability_matrix = np.concatenate(probability_batches, axis=0)

    expected_shape = (len(saved_validation_ids), len(LABEL_NAMES))

    if actual_matrix.shape != expected_shape:
        raise ValueError(
            f"Expected targets shape {expected_shape}, "
            f"got {actual_matrix.shape}"
        )

    if probability_matrix.shape != expected_shape:
        raise ValueError(
            f"Expected predictions shape {expected_shape}, "
            f"got {probability_matrix.shape}"
        )

    # --------------------------------------------------------
    # 5. Calculate per-label metrics
    # --------------------------------------------------------

    metrics_rows = []
    prediction_rows = []

    for label_index, label_name in enumerate(LABEL_NAMES):
        actual = actual_matrix[:, label_index].astype(np.int32)
        probabilities = probability_matrix[:, label_index]
        predicted = (probabilities >= THRESHOLD).astype(np.int32)

        metrics = calculate_metrics(
            actual=actual,
            probabilities=probabilities,
            threshold=THRESHOLD,
        )

        metrics_rows.append({
            "label": label_name,
            "threshold": THRESHOLD,
            **metrics,
        })

        for row_index, study_id in enumerate(saved_validation_ids):
            prediction_rows.append({
                "StudyInstanceUID": study_id,
                "label": label_name,
                "actual": int(actual[row_index]),
                "probability": float(probabilities[row_index]),
                "predicted": int(predicted[row_index]),
                "threshold": THRESHOLD,
            })

    # --------------------------------------------------------
    # 6. Save results without overwriting earlier evaluations
    # --------------------------------------------------------

    output_dir = create_evaluation_directory()

    metrics_path = output_dir / "per_label_metrics.csv"
    predictions_path = output_dir / "validation_predictions.csv"
    summary_path = output_dir / "evaluation_summary.txt"

    with metrics_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(metrics_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    with predictions_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(prediction_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    finite_losses = {
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_validation_loss": checkpoint["validation_loss"],
        "validation_studies": len(saved_validation_ids),
        "threshold": THRESHOLD,
    }

    with summary_path.open("w", encoding="utf-8") as file:
        file.write("RSNA Knee Model Evaluation\n")
        file.write("=" * 40 + "\n")

        for key, value in finite_losses.items():
            file.write(f"{key}: {value}\n")

        file.write("\nPer-label results:\n")

        for row in metrics_rows:
            file.write(
                f"{row['label']}: "
                f"positive={row['positive_count']}, "
                f"negative={row['negative_count']}, "
                f"sensitivity={row['sensitivity']:.4f}, "
                f"specificity={row['specificity']:.4f}, "
                f"precision={row['precision']:.4f}, "
                f"F1={row['f1']:.4f}, "
                f"AUC={row['roc_auc']:.4f}\n"
            )

    # --------------------------------------------------------
    # 7. Print a compact report
    # --------------------------------------------------------

    print()
    print("=" * 105)
    print("VALIDATION RESULTS")
    print("=" * 105)

    print(
        f"{'Label':<20}"
        f"{'Pos':>5}"
        f"{'Neg':>5}"
        f"{'Sens.':>10}"
        f"{'Spec.':>10}"
        f"{'Prec.':>10}"
        f"{'F1':>10}"
        f"{'AUC':>10}"
    )

    for row in metrics_rows:
        def fmt(value):
            return f"{value:.3f}" if math.isfinite(value) else "N/A"

        print(
            f"{row['label']:<20}"
            f"{row['positive_count']:>5}"
            f"{row['negative_count']:>5}"
            f"{fmt(row['sensitivity']):>10}"
            f"{fmt(row['specificity']):>10}"
            f"{fmt(row['precision']):>10}"
            f"{fmt(row['f1']):>10}"
            f"{fmt(row['roc_auc']):>10}"
        )

    print()
    print("EVALUATION COMPLETED")
    print(f"Output folder: {output_dir}")
    print(f"Metrics:       {metrics_path}")
    print(f"Predictions:   {predictions_path}")
    print(f"Summary:       {summary_path}")

    try:
        import sklearn  # noqa: F401
    except ImportError:
        print()
        print(
            "Note: scikit-learn is not installed, so ROC-AUC "
            "is reported as N/A. Install it in your project "
            "environment with: python -m pip install scikit-learn"
        )


if __name__ == "__main__":
    main()
