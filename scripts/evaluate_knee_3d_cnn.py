
from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ish_knee.dataset.knee_mri_dataset import KneeMRIDataset
from ish_knee.model.knee_3d_cnn import Knee3DCNN


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "outputs" / "processed_dataset_v1"
MANIFEST_PATH = DATASET_DIR / "manifest.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training_phase3d"
CHECKPOINT_PATH = OUTPUT_DIR / "knee_3d_cnn_best.pt"

METRICS_PATH = OUTPUT_DIR / "validation_metrics_v2.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "validation_predictions_v2.csv"

BATCH_SIZE = 1
THRESHOLD = 0.5

LABEL_NAMES = [
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
]


def calculate_auc(y_true, y_score):
    """
    Calculate ROC-AUC using positive-negative pairs.

    Returns None if the label has no positive or no
    negative examples in the validation set.
    """

    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    positive_scores = y_score[y_true == 1]
    negative_scores = y_score[y_true == 0]

    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return None

    comparisons = (
        positive_scores[:, None] - negative_scores[None, :]
    )

    return float(
        (np.sum(comparisons > 0) +
         0.5 * np.sum(comparisons == 0))
        / comparisons.size
    )


def calculate_metrics(y_true, y_prob):
    """
    Calculate per-label classification metrics.
    """

    y_pred = (y_prob >= THRESHOLD).astype(np.int64)

    results = []

    for index, label_name in enumerate(LABEL_NAMES):
        actual = y_true[:, index].astype(np.int64)
        predicted = y_pred[:, index]

        tp = int(np.sum((actual == 1) & (predicted == 1)))
        fp = int(np.sum((actual == 0) & (predicted == 1)))
        fn = int(np.sum((actual == 1) & (predicted == 0)))
        tn = int(np.sum((actual == 0) & (predicted == 0)))

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        auc = calculate_auc(actual, y_prob[:, index])

        results.append({
            "label": label_name,
            "positive_count": int(np.sum(actual == 1)),
            "negative_count": int(np.sum(actual == 0)),
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": auc,
        })

    return results


def main():
    # --------------------------------------------------
    # Validate required files
    # --------------------------------------------------

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {MANIFEST_PATH}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {CHECKPOINT_PATH}"
        )

    # Protect previous evaluation results.
    if METRICS_PATH.exists() or PREDICTIONS_PATH.exists():
        raise FileExistsError(
            "Evaluation output already exists. Rename the existing "
            "evaluation CSV files before running again."
        )

    # --------------------------------------------------
    # Load validation dataset
    # --------------------------------------------------

    dataset = KneeMRIDataset(
        manifest_path=MANIFEST_PATH,
        split="validation",
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("=" * 65)
    print("KNEE 3D CNN MODEL EVALUATION")
    print("=" * 65)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Validation studies: {len(dataset)}")
    print(f"Probability threshold: {THRESHOLD}")
    print()

    # --------------------------------------------------
    # Load trained model
    # --------------------------------------------------

    device = torch.device("cpu")

    model = Knee3DCNN().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(
        "Checkpoint validation BCE: "
        f"{checkpoint['validation_loss']:.6f}"
    )
    print()

    # --------------------------------------------------
    # Run validation predictions
    # --------------------------------------------------
    
    all_study_uids = []
    all_logits = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for study_uids, images, labels in loader:
            all_study_uids.extend(study_uids)

            images = images.to(device)
            labels = labels.to(device)

            sagittal = images[:, 0]
            coronal = images[:, 1]
            axial = images[:, 2]

            logits = model(
                sagittal,
                coronal,
                axial,
            )

            probabilities = torch.sigmoid(logits)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_probabilities.append(probabilities.cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    y_true = torch.cat(all_labels, dim=0).numpy()
    y_prob = torch.cat(all_probabilities, dim=0).numpy()

    if len(all_study_uids) != len(y_true):
        raise ValueError(
            "Study UID count does not match prediction count."
        )

    if len(set(all_study_uids)) != len(all_study_uids):
        raise ValueError(
            "Duplicate study UIDs found in validation predictions."
        )


    # Mean BCE across all studies and all 12 labels.
    validation_bce = F.binary_cross_entropy_with_logits(
        torch.from_numpy(logits),
        torch.from_numpy(y_true),
        reduction="mean",
    ).item()

    print(f"Validation BCE: {validation_bce:.6f}")
    print()

    # --------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------

    metrics = calculate_metrics(y_true, y_prob)

    print("-" * 105)
    print(
        f"{'Label':<20}"
        f"{'Pos':>5}"
        f"{'Neg':>5}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'ROC-AUC':>12}"
    )
    print("-" * 105)

    for result in metrics:
        auc = result["roc_auc"]
        auc_text = f"{auc:.3f}" if auc is not None else "N/A"

        print(
            f"{result['label']:<20}"
            f"{result['positive_count']:>5}"
            f"{result['negative_count']:>5}"
            f"{result['precision']:>12.3f}"
            f"{result['recall']:>10.3f}"
            f"{result['f1']:>10.3f}"
            f"{auc_text:>12}"
        )

    print("-" * 105)

    # --------------------------------------------------
    # Save per-label metrics
    # --------------------------------------------------

    with METRICS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(metrics[0].keys()),
        )
        writer.writeheader()
        writer.writerows(metrics)

    
    # --------------------------------------------------
    # Save predictions for every validation study
    # --------------------------------------------------

    prediction_fields = ["study_instance_uid"]

    for label_name in LABEL_NAMES:
        prediction_fields.extend([
            f"{label_name}_actual",
            f"{label_name}_probability",
        ])

    with PREDICTIONS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=prediction_fields,
        )
        writer.writeheader()

        for row_index, study_uid in enumerate(all_study_uids):
            output_row = {
                "study_instance_uid": study_uid
            }

            for label_index, label_name in enumerate(LABEL_NAMES):
                output_row[f"{label_name}_actual"] = int(
                    y_true[row_index, label_index]
                )
                output_row[f"{label_name}_probability"] = (
                    f"{y_prob[row_index, label_index]:.6f}"
                )

            writer.writerow(output_row)


    print()
    print("Evaluation completed.")
    print(f"Metrics:     {METRICS_PATH}")
    print(f"Predictions: {PREDICTIONS_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()
