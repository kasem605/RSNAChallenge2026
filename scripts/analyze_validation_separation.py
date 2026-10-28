
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation_phase3e_v2"
    / "validation_predictions.csv"
)

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


def calculate_auc(actual, probability):
    positive = probability[actual == 1]
    negative = probability[actual == 0]

    if len(positive) == 0 or len(negative) == 0:
        return np.nan

    comparisons = positive[:, None] - negative[None, :]

    return (
        np.sum(comparisons > 0)
        + 0.5 * np.sum(comparisons == 0)
    ) / comparisons.size


def main():
    if not PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(PREDICTIONS_PATH)

    df = pd.read_csv(PREDICTIONS_PATH)
    results = []

    for label in LABELS:
        actual = df[f"{label}_actual"].to_numpy(dtype=int)
        probability = df[
            f"{label}_probability"
        ].to_numpy(dtype=float)

        positive_scores = probability[actual == 1]
        negative_scores = probability[actual == 0]

        best_f1 = -1.0
        best_threshold = np.nan

        # Diagnostic threshold search on validation data.
        # Do not use this threshold as a final test-set
        # threshold without a separate validation strategy.
        for threshold in np.linspace(0.05, 0.95, 181):
            predicted = probability >= threshold

            tp = np.sum((actual == 1) & predicted)
            fp = np.sum((actual == 0) & predicted)
            fn = np.sum((actual == 1) & ~predicted)

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0

            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        results.append({
            "label": label,
            "positive_count": len(positive_scores),
            "negative_count": len(negative_scores),
            "positive_mean": (
                positive_scores.mean()
                if len(positive_scores) else np.nan
            ),
            "negative_mean": (
                negative_scores.mean()
                if len(negative_scores) else np.nan
            ),
            "mean_difference": (
                positive_scores.mean() - negative_scores.mean()
                if len(positive_scores) and len(negative_scores)
                else np.nan
            ),
            "roc_auc": calculate_auc(actual, probability),
            "best_validation_f1": best_f1,
            "best_validation_threshold": best_threshold,
        })

    result_df = pd.DataFrame(results)

    print("\nVALIDATION PROBABILITY SEPARATION")
    print(result_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))

    output_path = (
        PROJECT_ROOT
        / "outputs"
        / "evaluation_phase3e_v2"
        / "validation_separation.csv"
    )

    result_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
