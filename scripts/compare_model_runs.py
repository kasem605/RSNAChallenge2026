from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINE_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "evaluations"
    / "knee_eval_20261019_162011"
    / "validation_predictions.csv"
)

WEIGHTED_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "evaluations"
    / "knee_eval_20261019_174748"
    / "validation_predictions.csv"
)

THRESHOLD = 0.5

REQUIRED_COLUMNS = {
    "StudyInstanceUID",
    "label",
    "actual",
    "probability",
    "predicted",
    "threshold",
}


def load_predictions(path: Path, model_name: str) -> pd.DataFrame:
    """Load and validate a model's saved validation predictions."""

    if not path.exists():
        raise FileNotFoundError(f"{model_name} predictions not found: {path}")

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{model_name} CSV is missing columns: {sorted(missing)}"
        )

    df = df.copy()

    key_columns = ["StudyInstanceUID", "label"]

    if df[key_columns].isna().any().any():
        raise ValueError(f"{model_name} contains missing study/label keys.")

    if df.duplicated(key_columns).any():
        raise ValueError(f"{model_name} contains duplicate study/label rows.")

    df["actual"] = pd.to_numeric(df["actual"], errors="raise")
    df["probability"] = pd.to_numeric(df["probability"], errors="raise")

    if not df["actual"].isin([0, 1]).all():
        raise ValueError(f"{model_name} has non-binary actual labels.")

    if not np.isfinite(df["probability"]).all():
        raise ValueError(f"{model_name} has non-finite probabilities.")

    if not df["probability"].between(0, 1).all():
        raise ValueError(f"{model_name} probabilities must be between 0 and 1.")

    return df


def calculate_metrics(
    actual: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict:
    """Calculate per-label classification and ranking metrics."""

    predicted = (probability >= threshold).astype(int)

    positive_count = int(actual.sum())
    negative_count = int(len(actual) - positive_count)

    auc = (
        roc_auc_score(actual, probability)
        if positive_count > 0 and negative_count > 0
        else np.nan
    )

    average_precision = (
        average_precision_score(actual, probability)
        if positive_count > 0
        else np.nan
    )

    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())

    return {
        "N": len(actual),
        "Pos": positive_count,
        "Neg": negative_count,
        "Predicted_Pos": int(predicted.sum()),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Sensitivity": recall_score(
            actual, predicted, zero_division=0
        ),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "Precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "F1": f1_score(actual, predicted, zero_division=0),
        "ROC_AUC": auc,
        "Average_Precision": average_precision,
        "Mean_Probability": float(probability.mean()),
    }


def main() -> None:
    baseline = load_predictions(BASELINE_FILE, "Baseline")
    weighted = load_predictions(WEIGHTED_FILE, "Weighted")

    keys = ["StudyInstanceUID", "label"]

    baseline_keys = set(map(tuple, baseline[keys].to_numpy()))
    weighted_keys = set(map(tuple, weighted[keys].to_numpy()))

    if baseline_keys != weighted_keys:
        missing_from_weighted = baseline_keys - weighted_keys
        missing_from_baseline = weighted_keys - baseline_keys

        raise ValueError(
            "The two evaluations do not contain identical study/label keys. "
            f"Missing from weighted: {len(missing_from_weighted)}; "
            f"missing from baseline: {len(missing_from_baseline)}."
        )

    comparison = baseline[
        keys + ["actual", "probability"]
    ].merge(
        weighted[keys + ["actual", "probability"]],
        on=keys,
        how="inner",
        suffixes=("_baseline", "_weighted"),
        validate="one_to_one",
    )

    if not np.array_equal(
        comparison["actual_baseline"].to_numpy(),
        comparison["actual_weighted"].to_numpy(),
    ):
        raise ValueError(
            "Actual labels differ between the baseline and weighted files."
        )

    comparison["probability_change"] = (
        comparison["probability_weighted"]
        - comparison["probability_baseline"]
    )

    comparison["absolute_probability_change"] = (
        comparison["probability_change"].abs()
    )

    label_results = []

    for label in sorted(comparison["label"].unique()):
        subset = comparison[comparison["label"] == label]

        actual = subset["actual_baseline"].to_numpy(dtype=int)
        baseline_prob = subset["probability_baseline"].to_numpy(dtype=float)
        weighted_prob = subset["probability_weighted"].to_numpy(dtype=float)

        baseline_metrics = calculate_metrics(
            actual, baseline_prob, THRESHOLD
        )
        weighted_metrics = calculate_metrics(
            actual, weighted_prob, THRESHOLD
        )

        row = {"Label": label}

        for metric, value in baseline_metrics.items():
            row[f"Baseline_{metric}"] = value

        for metric, value in weighted_metrics.items():
            row[f"Weighted_{metric}"] = value

        row["AUC_Change"] = (
            weighted_metrics["ROC_AUC"] - baseline_metrics["ROC_AUC"]
            if np.isfinite(baseline_metrics["ROC_AUC"])
            and np.isfinite(weighted_metrics["ROC_AUC"])
            else np.nan
        )

        row["AP_Change"] = (
            weighted_metrics["Average_Precision"]
            - baseline_metrics["Average_Precision"]
            if np.isfinite(baseline_metrics["Average_Precision"])
            and np.isfinite(weighted_metrics["Average_Precision"])
            else np.nan
        )

        row["Mean_Probability_Change"] = float(
            (weighted_prob - baseline_prob).mean()
        )

        row["Mean_Absolute_Probability_Change"] = float(
            np.abs(weighted_prob - baseline_prob).mean()
        )

        label_results.append(row)

    results = pd.DataFrame(label_results)

    output_base = PROJECT_ROOT / "artifacts" / "evaluations"
    output_base.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / f"model_comparison_{timestamp}"
    output_dir.mkdir()

    metrics_path = output_dir / "model_comparison.csv"
    paired_path = output_dir / "paired_probabilities.csv"
    summary_path = output_dir / "comparison_summary.txt"

    results.to_csv(metrics_path, index=False)
    comparison.to_csv(paired_path, index=False)

    auc_changes = results["AUC_Change"].dropna()

    lines = [
        "BASELINE VS WEIGHTED MODEL COMPARISON",
        "=" * 70,
        f"Baseline predictions: {BASELINE_FILE}",
        f"Weighted predictions: {WEIGHTED_FILE}",
        f"Matched study-label rows: {len(comparison)}",
        f"Unique validation studies: {comparison['StudyInstanceUID'].nunique()}",
        f"Labels: {comparison['label'].nunique()}",
        f"Classification threshold: {THRESHOLD}",
        "",
        "Per-label metrics are saved in model_comparison.csv.",
        "Paired probabilities are saved in paired_probabilities.csv.",
        "",
        "ROC-AUC changes (weighted minus baseline):",
        f"Labels with higher AUC: {int((auc_changes > 0).sum())}",
        f"Labels with lower AUC: {int((auc_changes < 0).sum())}",
        f"Labels with unchanged AUC: {int((auc_changes == 0).sum())}",
        "",
        "A positive AUC change indicates improved ranking on this validation set.",
        "It does not establish clinical usefulness or statistical significance.",
    ]

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nMODEL COMPARISON")
    print("=" * 110)

    display_columns = [
        "Label",
        "Baseline_ROC_AUC",
        "Weighted_ROC_AUC",
        "AUC_Change",
        "Baseline_Average_Precision",
        "Weighted_Average_Precision",
        "Baseline_Sensitivity",
        "Weighted_Sensitivity",
        "Baseline_Specificity",
        "Weighted_Specificity",
    ]

    print(results[display_columns].round(3).to_string(index=False))

    print("\nCOMPARISON COMPLETED")
    print(f"Output folder: {output_dir}")
    print(f"Metrics:       {metrics_path}")
    print(f"Paired data:   {paired_path}")
    print(f"Summary:       {summary_path}")


if __name__ == "__main__":
    main()
