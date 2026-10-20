
import csv
import math
from pathlib import Path
from collections import defaultdict

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "evaluations"
    / "knee_eval_20261019_162011"
    / "validation_predictions.csv"
)

THRESHOLD = 0.5


def main() -> None:
    if not PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            f"Predictions file not found:\n{PREDICTIONS_PATH}"
        )

    rows_by_label = defaultdict(list)

    with PREDICTIONS_PATH.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "StudyInstanceUID",
            "label",
            "actual",
            "probability",
            "predicted",
            "threshold",
        }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Missing CSV columns: "
                f"{required - set(reader.fieldnames or [])}"
            )

        for row in reader:
            rows_by_label[row["label"]].append({
                "study_id": row["StudyInstanceUID"],
                "actual": int(row["actual"]),
                "probability": float(row["probability"]),
                "predicted": int(row["predicted"]),
            })

    if not rows_by_label:
        raise ValueError("No predictions found.")

    print("=" * 115)
    print("KNEE MODEL PROBABILITY DIAGNOSTICS")
    print("=" * 115)

    print(
        f"{'Label':<20}"
        f"{'N':>5}"
        f"{'Mean':>9}"
        f"{'Min':>9}"
        f"{'Max':>9}"
        f"{'<0.5':>9}"
        f"{'>=0.5':>9}"
        f"{'Pos mean':>11}"
        f"{'Neg mean':>11}"
        f"{'Gap':>9}"
    )

    diagnostic_rows = []

    for label, rows in sorted(rows_by_label.items()):
        probabilities = np.array(
            [row["probability"] for row in rows],
            dtype=float,
        )
        actual = np.array(
            [row["actual"] for row in rows],
            dtype=int,
        )

        if not np.isfinite(probabilities).all():
            raise ValueError(
                f"{label} contains non-finite probabilities."
            )

        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError(
                f"{label} contains probabilities outside [0, 1]."
            )

        if not np.isin(actual, [0, 1]).all():
            raise ValueError(
                f"{label} contains non-binary actual labels."
            )

        positive_probs = probabilities[actual == 1]
        negative_probs = probabilities[actual == 0]

        positive_mean = (
            float(positive_probs.mean())
            if len(positive_probs)
            else float("nan")
        )
        negative_mean = (
            float(negative_probs.mean())
            if len(negative_probs)
            else float("nan")
        )

        gap = (
            positive_mean - negative_mean
            if math.isfinite(positive_mean)
            and math.isfinite(negative_mean)
            else float("nan")
        )

        below = int(np.sum(probabilities < THRESHOLD))
        above = int(np.sum(probabilities >= THRESHOLD))

        diagnostic_rows.append({
            "label": label,
            "count": len(rows),
            "mean_probability": float(probabilities.mean()),
            "minimum_probability": float(probabilities.min()),
            "maximum_probability": float(probabilities.max()),
            "below_threshold": below,
            "at_or_above_threshold": above,
            "positive_mean_probability": positive_mean,
            "negative_mean_probability": negative_mean,
            "positive_minus_negative_mean": gap,
        })

        def fmt(value):
            return f"{value:.4f}" if math.isfinite(value) else "N/A"

        print(
            f"{label:<20}"
            f"{len(rows):>5}"
            f"{probabilities.mean():>9.4f}"
            f"{probabilities.min():>9.4f}"
            f"{probabilities.max():>9.4f}"
            f"{below:>9}"
            f"{above:>9}"
            f"{fmt(positive_mean):>11}"
            f"{fmt(negative_mean):>11}"
            f"{fmt(gap):>9}"
        )

    output_path = (
        PREDICTIONS_PATH.parent
        / "probability_diagnostics.csv"
    )

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(diagnostic_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    print()
    print(f"Diagnostic report saved to:\n{output_path}")

    print()
    print("Interpretation guide:")
    print("- Positive mean > negative mean: potentially useful ranking signal.")
    print("- Positive mean < negative mean: ranking may be reversed or noisy.")
    print("- All probabilities on one side of 0.5: threshold issue is possible.")
    print("- These mean differences are descriptive, not proof of model quality.")


if __name__ == "__main__":
    main()
