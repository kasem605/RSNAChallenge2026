
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ish_knee.dataset.knee_mri_dataset import KneeMRIDataset
from ish_knee.model.knee_3d_cnn import Knee3DCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT / "outputs" / "processed_dataset_v1" / "manifest.csv"
)

CHECKPOINTS = {
    "unweighted": (
        PROJECT_ROOT / "outputs" / "weighted_bce_comparison"
        / "unweighted" / "best_model.pt"
    ),
    "weighted": (
        PROJECT_ROOT / "outputs" / "weighted_bce_comparison"
        / "weighted" / "best_model.pt"
    ),
}

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "bce_checkpoint_evaluation_v2"

LABEL_NAMES = [
    "acl", "mcl", "medial_meniscus", "lateral_meniscus",
    "medial_oa", "lateral_oa", "pf_oa", "effusion",
    "synovitis", "bakers", "contusion", "fracture",
]


def evaluate_checkpoint(checkpoint_path, loader, experiment):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model = Knee3DCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probabilities_all = []
    labels_all = []
    total_loss = 0.0
    total_elements = 0
    study_uids_all = []

    with torch.no_grad():
        for study_uids, images, labels in loader:
            images = images.float()
            labels = labels.float()

            study_uids_all.extend(study_uids)
            logits = model(
                images[:, 0],
                images[:, 1],
                images[:, 2],
            )

            # Always use ordinary BCE for this comparison.
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, reduction="sum"
            )

            total_loss += loss.item()
            total_elements += labels.numel()

            probabilities_all.append(
                torch.sigmoid(logits).numpy()
            )
            labels_all.append(labels.numpy())

    probabilities = np.concatenate(probabilities_all, axis=0)
    actual = np.concatenate(labels_all, axis=0)

    if not np.isfinite(actual).all():
        raise ValueError(
            "Validation labels contain missing or non-finite values."
        )

    if not np.isin(actual, [0.0, 1.0]).all():
        raise ValueError(
            "Validation labels must contain only 0 and 1."
        )

    ordinary_bce = total_loss / total_elements
    metric_rows = []
    prediction_rows = []

    for i, label_name in enumerate(LABEL_NAMES):
        p = probabilities[:, i]
        y = actual[:, i].astype(int)
        predicted = (p >= 0.5).astype(int)

        tp = int(((predicted == 1) & (y == 1)).sum())
        fp = int(((predicted == 1) & (y == 0)).sum())
        fn = int(((predicted == 0) & (y == 1)).sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        metric_rows.append({
            "experiment": experiment,
            "label": label_name,
            "actual_positive": int(y.sum()),
            "predicted_positive": int(predicted.sum()),
            "probability_mean": float(p.mean()),
            "probability_std": float(p.std()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    if len(study_uids_all) != len(actual):
        raise ValueError(
            "Study ID count does not match prediction count."
        )

    if len(set(study_uids_all)) != len(study_uids_all):
        raise ValueError("Duplicate study IDs found in predictions.")

    for study_index, study_uid in enumerate(study_uids_all):
        row = {
            "study_instance_uid": study_uid,
            "split": "validation",
            "dataset_index": study_index,
        }

        for i, label_name in enumerate(LABEL_NAMES):
            row[f"actual_{label_name}"] = int(
                actual[study_index, i]
            )
            row[f"prob_{label_name}"] = float(
                probabilities[study_index, i]
            )

        prediction_rows.append(row)

    summary = {
        "experiment": experiment,
        "checkpoint": str(checkpoint_path),
        "training_objective_best_epoch": checkpoint["best_epoch"],
        "ordinary_validation_bce": ordinary_bce,
    }

    return summary, metric_rows, prediction_rows


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    for name, path in CHECKPOINTS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {name} checkpoint: {path}"
            )

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {OUTPUT_DIR}. "
            "Choose a new output directory to preserve existing results."
        )

    dataset = KneeMRIDataset(
        manifest_path=MANIFEST_PATH,
        split="validation",
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    summaries = []
    all_metrics = []

    for experiment, checkpoint_path in CHECKPOINTS.items():
        print(f"\nEvaluating {experiment}: {checkpoint_path}")

        summary, metrics, predictions = evaluate_checkpoint(
            checkpoint_path, loader, experiment
        )

        summaries.append(summary)
        all_metrics.extend(metrics)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(predictions).to_csv(
            OUTPUT_DIR / f"{experiment}_predictions.csv",
            index=False,
        )

    summary_df = pd.DataFrame(summaries)
    metrics_df = pd.DataFrame(all_metrics)

    summary_df.to_csv(
        OUTPUT_DIR / "checkpoint_summary.csv", index=False
    )
    metrics_df.to_csv(
        OUTPUT_DIR / "per_label_metrics.csv", index=False
    )

    print("\nCHECKPOINT SUMMARY — ORDINARY BCE")
    print(
        summary_df[
            ["experiment", "training_objective_best_epoch",
             "ordinary_validation_bce"]
        ].to_string(index=False)
    )

    print("\nPER-LABEL METRICS")
    print(metrics_df.to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
