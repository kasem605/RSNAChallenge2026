
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from ish_knee.dataset.knee_mri_dataset import KneeMRIDataset
from ish_knee.model.knee_3d_cnn import Knee3DCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT / "outputs" / "processed_dataset_v1" / "manifest.csv"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "training_lr001_30epochs"
    / "lr_0.001"
    / "best_model.pt"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "per_label_discrimination"

LABELS = [
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


def load_checkpoint(model, checkpoint_path, device):
    """Load either a raw state dictionary or a checkpoint dictionary."""

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise TypeError(
            "Unexpected checkpoint format. "
            "Expected a state dictionary or checkpoint dictionary."
        )

    model.load_state_dict(state_dict)
    return model



def collect_predictions(model, manifest_path, split, device):
    """Collect study IDs, labels, and probabilities for a split."""

    dataset = KneeMRIDataset(
        manifest_path=str(manifest_path),
        split=split,
    )

    study_uids = []
    actual_all = []
    probability_all = []

    model.eval()

    with torch.no_grad():
        for index in range(len(dataset)):
            study_uid, images, labels = dataset[index]

            images = images.unsqueeze(0).to(device)
            labels = labels.detach().cpu().numpy().reshape(-1)

            sagittal = images[:, 0]
            coronal = images[:, 1]
            axial = images[:, 2]

            logits = model(sagittal, coronal, axial)
            probabilities = torch.sigmoid(logits)

            study_uids.append(study_uid)
            actual_all.append(labels)
            probability_all.append(
                probabilities.squeeze(0).cpu().numpy()
            )

    if not actual_all:
        raise ValueError(f"No studies found for split: {split}")

    return (
        study_uids,
        np.asarray(actual_all, dtype=np.float64),
        np.asarray(probability_all, dtype=np.float64),
    )



def safe_metric(metric_function, actual, scores):
    """Return NaN when a metric cannot be computed."""

    try:
        return float(metric_function(actual, scores))
    except (ValueError, ZeroDivisionError):
        return float("nan")


def evaluate_split(split, actual, probabilities, train_prevalence):
    rows = []

    for index, label_name in enumerate(LABELS):
        y_true = actual[:, index].astype(int)
        y_score = probabilities[:, index]

        positives = int(y_true.sum())
        negatives = int(len(y_true) - positives)

        if positives > 0 and negatives > 0:
            auc = safe_metric(
                roc_auc_score, y_true, y_score
            )
        else:
            auc = float("nan")

        if positives > 0:
            ap = safe_metric(
                average_precision_score, y_true, y_score
            )
        else:
            ap = float("nan")

        # Baseline AP: assign every study the training prevalence.
        # This is a reference, not a trained classifier.
        prevalence_baseline_ap = float(train_prevalence[index])

        rows.append({
            "split": split,
            "label": label_name,
            "study_count": len(y_true),
            "positive_count": positives,
            "negative_count": negatives,
            "observed_prevalence": float(y_true.mean()),
            "train_prevalence_baseline": prevalence_baseline_ap,
            "roc_auc": auc,
            "average_precision": ap,
            "ap_minus_prevalence_baseline": (
                ap - prevalence_baseline_ap
                if np.isfinite(ap)
                else float("nan")
            ),
            "probability_mean": float(y_score.mean()),
            "probability_std": float(y_score.std()),
            "probability_min": float(y_score.min()),
            "probability_max": float(y_score.max()),
        })

    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)

    #--------------------------------------------------------------------
    # check required columns
    # -------------------------------------------------------------------
    required_columns = [
        "split",
        *[f"label_{label}" for label in LABELS],
    ]
    missing = [
        column for column in required_columns
        if column not in manifest.columns
    ]

    if missing:
        raise ValueError(
            f"Missing manifest columns: {missing}"
        )

    #--------------------------------------------------------------------
    # check duplicate studies
    # -------------------------------------------------------------------
    duplicates = manifest[
        manifest["study_instance_uid"].duplicated(keep=False)
    ]

    if not duplicates.empty:
        print("WARNING: Duplicate study IDs found:")
        print(duplicates[["study_instance_uid", "split"]])
    else:
        print("PASS: No duplicate study IDs.")

        train_manifest = manifest[
            manifest["split"] == "train"
        ]

    #--------------------------------------------------------------------
    # Check train/validation overlap
    # -------------------------------------------------------------------
    train_ids = set(
        manifest.loc[manifest["split"] == "train", "study_instance_uid"]
    )
    val_ids = set(
        manifest.loc[manifest["split"] == "validation", "study_instance_uid"]
    )

    overlap = train_ids.intersection(val_ids)

    if overlap:
        raise ValueError(f"Train/validation leakage: {len(overlap)} studies overlap.")

    print(f"Training studies: {len(train_ids)}")
    print(f"Validation studies: {len(val_ids)}")
    print("PASS: No train/validation overlap.")

    # NaN labels are unknown, not negative.
    train_labels = train_manifest[
        [f"label_{label}" for label in LABELS]
    ].to_numpy(dtype=np.float64)

    if np.isnan(train_labels).any():
        raise ValueError(
            "Training manifest contains unknown labels. "
            "Check that the manifest contains only fully labeled "
            "studies in its train and validation splits."
        )

    train_prevalence = train_labels.mean(axis=0)

    model = Knee3DCNN().to(device)
    load_checkpoint(model, CHECKPOINT_PATH, device)

    all_rows = []
    collected = {}

    for split in ["train", "validation"]:
        print(f"\nEvaluating {split} split...")

        study_uids, actual, probabilities = collect_predictions(
            model,
            MANIFEST_PATH,
            split,
            device,
        )

        if actual.shape[1] != len(LABELS):
            raise ValueError(
                f"Expected {len(LABELS)} labels, "
                f"received {actual.shape[1]}"
            )

        if not np.isfinite(probabilities).all():
            raise ValueError(
                f"Non-finite probabilities found in {split}"
            )

        rows = evaluate_split(
            split,
            actual,
            probabilities,
            train_prevalence,
        )
        all_rows.extend(rows)

        collected[split] = {
            "study_count": int(len(actual)),
            "study_uids": study_uids,
            "actual": actual,
            "probabilities": probabilities,
        }

        print(f"Studies evaluated: {len(actual)}")

    results = pd.DataFrame(all_rows)

    results_path = OUTPUT_DIR / "per_label_metrics.csv"
    results.to_csv(results_path, index=False)

    # Save per-study probabilities with the original study IDs.
    prediction_rows = []

    # Load the manifest to map each split's dataset index to its study UID.
    for split, values in collected.items():
        if len(values["study_uids"]) != values["study_count"]:
            raise ValueError(f"{split}: study ID count does not match predictions.")

        if len(set(values["study_uids"])) != values["study_count"]:
            raise ValueError(f"{split}: duplicate study IDs in predictions.")

        for study_index, study_uid in enumerate(values["study_uids"]):
            row = {
                "study_instance_uid": study_uid,
                "split": split,
                "study_index": study_index,
            }

            for label_index, label_name in enumerate(LABELS):
                row[f"actual_{label_name}"] = int(
                    values["actual"][study_index, label_index]
                )
                row[f"probability_{label_name}"] = float(
                    values["probabilities"][study_index, label_index]
                )

            prediction_rows.append(row)

        predictions_path = OUTPUT_DIR / "study_predictions.csv"
        predictions_df = pd.DataFrame(prediction_rows)

        if predictions_df["study_instance_uid"].duplicated().any():
            raise ValueError("Duplicate study IDs in prediction output.")

        predictions_df.to_csv(predictions_path, index=False)

        print(f"Saved {len(predictions_df)} study predictions.")
        print(predictions_df[
            ["study_instance_uid", "split", "study_index"]
        ].head())


        predictions_path = OUTPUT_DIR / "study_predictions.csv"

        predictions_df = pd.DataFrame(prediction_rows)

        if predictions_df["study_instance_uid"].duplicated().any():
            raise ValueError(
                "Duplicate study IDs found in the prediction output."
            )

        predictions_df.to_csv(predictions_path, index=False)

        print(f"Saved {len(predictions_df)} study predictions.")
        print("Study IDs included in prediction output.")
        print(predictions_df[
            ["study_instance_uid", "split", "study_index"]
        ].head())

        summary = {
            "checkpoint": str(CHECKPOINT_PATH),
            "manifest": str(MANIFEST_PATH),
            "device": str(device),
            "label_order": LABELS,
            "train_studies": collected["train"]["study_count"],
            "validation_studies": collected["validation"]["study_count"],
            "train_prevalence": {
                label: float(train_prevalence[index])
                for index, label in enumerate(LABELS)
            },
            "note": (
                "Validation metrics are uncertain because the validation "
                "set is small. ROC-AUC is undefined if a split contains "
                "only one class for a label. AP is undefined here when "
                "there are no positive examples."
            ),
        }

    summary_path = OUTPUT_DIR / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    display_columns = [
        "split",
        "label",
        "positive_count",
        "negative_count",
        "roc_auc",
        "average_precision",
        "train_prevalence_baseline",
        "ap_minus_prevalence_baseline",
    ]

    print("\nPER-LABEL DISCRIMINATION RESULTS")
    print("=" * 100)
    print(
        results[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nFiles saved:")
    print(f"  {results_path}")
    print(f"  {predictions_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
