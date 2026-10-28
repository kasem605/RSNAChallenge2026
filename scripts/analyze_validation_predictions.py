
from pathlib import Path
import csv

import numpy as np


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "training_phase3d"
    / "validation_predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training_phase3d"

PROBABILITY_SUMMARY_PATH = (
    OUTPUT_DIR / "validation_probability_summary.csv"
)

THRESHOLD_METRICS_PATH = (
    OUTPUT_DIR / "validation_threshold_metrics.csv"
)

THRESHOLDS = [0.2, 0.3, 0.4, 0.5]

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


def load_predictions(path):
    """Load actual labels and predicted probabilities."""

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("The prediction file contains no rows.")

    actual = []
    probabilities = []

    for label in LABEL_NAMES:
        actual_column = f"{label}_actual"
        probability_column = f"{label}_probability"

        for column in [actual_column, probability_column]:
            if column not in rows[0]:
                raise ValueError(
                    f"Missing column in prediction CSV: {column}"
                )

        actual_values = [
            int(row[actual_column]) for row in rows
        ]

        probability_values = [
            float(row[probability_column]) for row in rows
        ]

        if not all(value in (0, 1) for value in actual_values):
            raise ValueError(
                f"Actual labels for {label} must be binary."
            )

        if not np.isfinite(probability_values).all():
            raise ValueError(
                f"Invalid probability found for {label}."
            )

        if not all(
            0.0 <= value <= 1.0
            for value in probability_values
        ):
            raise ValueError(
                f"Probabilities for {label} must be between 0 and 1."
            )

        actual.append(actual_values)
        probabilities.append(probability_values)

    return np.asarray(actual).T, np.asarray(probabilities).T


def calculate_metrics(y_true, y_prob, threshold):
    """Calculate threshold-based classification metrics."""

    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "predicted_positive": int(np.sum(y_pred)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def write_csv(path, rows, fieldnames):
    """Create a new CSV without overwriting existing results."""

    with path.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("=" * 72)
    print("VALIDATION PREDICTION ANALYSIS")
    print("=" * 72)

    y_true, y_prob = load_predictions(INPUT_PATH)

    print(f"Validation studies: {len(y_true)}")
    print(f"Labels: {len(LABEL_NAMES)}")
    print(f"Input: {INPUT_PATH}")
    print()

    # --------------------------------------------------
    # Part 1: Probability distributions
    # --------------------------------------------------

    summary_rows = []

    print("PROBABILITY DISTRIBUTION")
    print("-" * 100)
    print(
        f"{'Label':<20}"
        f"{'Min':>9}"
        f"{'Mean':>9}"
        f"{'Max':>9}"
        f"{'<0.5':>9}"
        f"{'>=0.5':>9}"
        f"{'Actual+':>10}"
    )
    print("-" * 100)

    for index, label in enumerate(LABEL_NAMES):
        actual = y_true[:, index]
        prob = y_prob[:, index]

        row = {
            "label": label,
            "actual_positive": int(np.sum(actual == 1)),
            "actual_negative": int(np.sum(actual == 0)),
            "probability_min": float(np.min(prob)),
            "probability_mean": float(np.mean(prob)),
            "probability_max": float(np.max(prob)),
            "probability_below_0_5": int(np.sum(prob < 0.5)),
            "probability_at_least_0_5": int(np.sum(prob >= 0.5)),
        }

        summary_rows.append(row)

        print(
            f"{label:<20}"
            f"{row['probability_min']:>9.3f}"
            f"{row['probability_mean']:>9.3f}"
            f"{row['probability_max']:>9.3f}"
            f"{row['probability_below_0_5']:>9}"
            f"{row['probability_at_least_0_5']:>9}"
            f"{row['actual_positive']:>10}"
        )

    print("-" * 100)

    # --------------------------------------------------
    # Part 2: Compare thresholds
    # --------------------------------------------------

    threshold_rows = []

    print()
    print("THRESHOLD COMPARISON")
    print("-" * 100)
    print(
        f"{'Label':<20}"
        f"{'Threshold':>10}"
        f"{'Predicted+':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
    )
    print("-" * 100)

    for index, label in enumerate(LABEL_NAMES):
        for threshold in THRESHOLDS:
            result = calculate_metrics(
                y_true[:, index],
                y_prob[:, index],
                threshold,
            )

            result["label"] = label
            threshold_rows.append(result)

            print(
                f"{label:<20}"
                f"{threshold:>10.2f}"
                f"{result['predicted_positive']:>12}"
                f"{result['precision']:>12.3f}"
                f"{result['recall']:>10.3f}"
                f"{result['f1']:>10.3f}"
            )

    print("-" * 100)

    # --------------------------------------------------
    # Part 3: Save diagnostic files
    # --------------------------------------------------

    summary_fields = [
        "label",
        "actual_positive",
        "actual_negative",
        "probability_min",
        "probability_mean",
        "probability_max",
        "probability_below_0_5",
        "probability_at_least_0_5",
    ]

    threshold_fields = [
        "label",
        "threshold",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
        "predicted_positive",
        "precision",
        "recall",
        "f1",
    ]

    write_csv(
        PROBABILITY_SUMMARY_PATH,
        summary_rows,
        summary_fields,
    )

    write_csv(
        THRESHOLD_METRICS_PATH,
        threshold_rows,
        threshold_fields,
    )

    print()
    print("Analysis completed successfully.")
    print(f"Probability summary: {PROBABILITY_SUMMARY_PATH}")
    print(f"Threshold metrics:   {THRESHOLD_METRICS_PATH}")
    print()
    print(
        "These threshold comparisons are diagnostic only. "
        "They do not establish generalization performance."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
