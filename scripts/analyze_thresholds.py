
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


# Project root is the parent of the scripts folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINE = (
    PROJECT_ROOT
    / "artifacts/evaluations/knee_eval_20261019_162011/validation_predictions.csv"
)

WEIGHTED = (
    PROJECT_ROOT
    / "artifacts/evaluations/knee_eval_20261019_174748/validation_predictions.csv"
)

OUTPUT_ROOT = PROJECT_ROOT / "artifacts/evaluations"

THRESHOLDS = np.round(np.arange(0.20, 0.701, 0.05), 2)

LABELS = [
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

KEYS = ["StudyInstanceUID", "label"]
REQUIRED = KEYS + ["actual", "probability"]


def load_predictions(path: Path, model_name: str) -> pd.DataFrame:
    """Load and validate saved predictions."""

    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} predictions not found:\n{path}\n"
            "Check the paths at the top of this script."
        )

    df = pd.read_csv(path)

    missing = [column for column in REQUIRED if column not in df.columns]
    if missing:
        raise ValueError(
            f"{model_name} file is missing columns: {missing}"
        )

    df = df[REQUIRED].copy()
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    df["label"] = df["label"].astype(str)
    df["actual"] = pd.to_numeric(
        df["actual"], errors="raise"
    ).astype(int)
    df["probability"] = pd.to_numeric(
        df["probability"], errors="raise"
    )

    if df.duplicated(KEYS).any():
        raise ValueError(
            f"{model_name} contains duplicate study/label rows."
        )

    if not df["actual"].isin([0, 1]).all():
        raise ValueError("'actual' must contain only 0 or 1.")

    if not df["probability"].between(0, 1).all():
        raise ValueError("Probabilities must be between 0 and 1.")

    return df.sort_values(KEYS).reset_index(drop=True)


def verify_same_records(
    baseline: pd.DataFrame,
    weighted: pd.DataFrame,
) -> None:
    """Confirm both models use identical validation targets."""

    if not baseline[KEYS].equals(weighted[KEYS]):
        raise ValueError(
            "Baseline and weighted files have different study/label keys."
        )

    if not np.array_equal(
        baseline["actual"].to_numpy(),
        weighted["actual"].to_numpy(),
    ):
        raise ValueError(
            "Baseline and weighted files have different actual labels."
        )


def calculate_auc(y, probabilities):
    """AUC requires at least one positive and one negative."""

    if len(np.unique(y)) < 2:
        return float("nan")

    return float(roc_auc_score(y, probabilities))


def analyze_thresholds(
    df: pd.DataFrame,
    model_name: str,
):
    """Calculate per-label metrics at each threshold."""

    detail_rows = []
    summary_rows = []

    for threshold in THRESHOLDS:
        sensitivities = []
        specificities = []
        precisions = []
        f1_scores = []
        total_predicted = 0

        for label in LABELS:
            part = df[df["label"] == label]

            if part.empty:
                continue

            y = part["actual"].to_numpy(dtype=int)
            probabilities = part["probability"].to_numpy(dtype=float)

            # Recalculate predictions at this threshold.
            predicted = (probabilities >= threshold).astype(int)

            tp = int(((y == 1) & (predicted == 1)).sum())
            fp = int(((y == 0) & (predicted == 1)).sum())
            tn = int(((y == 0) & (predicted == 0)).sum())
            fn = int(((y == 1) & (predicted == 0)).sum())

            positives = int((y == 1).sum())
            negatives = int((y == 0).sum())
            predicted_positives = int(predicted.sum())

            sensitivity = (
                tp / positives if positives else float("nan")
            )
            specificity = (
                tn / negatives if negatives else float("nan")
            )
            precision = (
                tp / predicted_positives
                if predicted_positives
                else 0.0
            )
            f1 = float(
                f1_score(y, predicted, zero_division=0)
            )

            auc = calculate_auc(y, probabilities)
            average_precision = (
                float(average_precision_score(y, probabilities))
                if positives
                else float("nan")
            )

            total_predicted += predicted_positives

            if np.isfinite(sensitivity):
                sensitivities.append(sensitivity)

            if np.isfinite(specificity):
                specificities.append(specificity)

            precisions.append(precision)
            f1_scores.append(f1)

            detail_rows.append({
                "model": model_name,
                "threshold": float(threshold),
                "label": label,
                "study_count": len(part),
                "actual_positive": positives,
                "actual_negative": negatives,
                "predicted_positive": predicted_positives,
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "precision": precision,
                "f1": f1,
                "roc_auc": auc,
                "average_precision": average_precision,
                "mean_probability": float(probabilities.mean()),
            })

        summary_rows.append({
            "model": model_name,
            "threshold": float(threshold),
            "labels_evaluated": len(f1_scores),
            "macro_sensitivity": (
                float(np.mean(sensitivities))
                if sensitivities else float("nan")
            ),
            "macro_specificity": (
                float(np.mean(specificities))
                if specificities else float("nan")
            ),
            "macro_precision": (
                float(np.mean(precisions))
                if precisions else float("nan")
            ),
            "macro_f1": (
                float(np.mean(f1_scores))
                if f1_scores else float("nan")
            ),
            "total_predicted_positives": total_predicted,
        })

    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def main():
    print("=" * 72)
    print("KNEE MODEL THRESHOLD ANALYSIS")
    print("=" * 72)

    baseline = load_predictions(BASELINE, "Baseline")
    weighted = load_predictions(WEIGHTED, "Weighted")

    verify_same_records(baseline, weighted)

    print(
        f"Validation studies: "
        f"{baseline['StudyInstanceUID'].nunique()}"
    )
    print(f"Study/label rows per model: {len(baseline)}")
    print("Thresholds:", ", ".join(f"{x:.2f}" for x in THRESHOLDS))

    baseline_detail, baseline_summary = analyze_thresholds(
        baseline, "baseline"
    )
    weighted_detail, weighted_summary = analyze_thresholds(
        weighted, "weighted"
    )

    details = pd.concat(
        [baseline_detail, weighted_detail],
        ignore_index=True,
    )
    summaries = pd.concat(
        [baseline_summary, weighted_summary],
        ignore_index=True,
    )

    # Always create a new folder to preserve previous results.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"threshold_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    detail_path = output_dir / "threshold_metrics_by_label.csv"
    summary_path = output_dir / "threshold_summary.csv"

    details.to_csv(detail_path, index=False)
    summaries.to_csv(summary_path, index=False)

    print("\nSUMMARY BY MODEL AND THRESHOLD")
    print(summaries[[
        "model",
        "threshold",
        "macro_sensitivity",
        "macro_specificity",
        "macro_precision",
        "macro_f1",
        "total_predicted_positives",
    ]].to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    ))

    print("\nFiles saved:")
    print(f"Detailed metrics: {detail_path}")
    print(f"Threshold summary: {summary_path}")

    print("\nImportant:")
    print("- AUC and average precision do not change with threshold.")
    print("- Lower thresholds generally predict more positives.")
    print("- There are only 17 validation studies; results are exploratory.")


if __name__ == "__main__":
    main()
