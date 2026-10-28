
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINE_FILE = (
    PROJECT_ROOT
    / "artifacts/evaluations/knee_eval_20261019_162011/validation_predictions.csv"
)

WEIGHTED_FILE = (
    PROJECT_ROOT
    / "artifacts/evaluations/knee_eval_20261019_174748/validation_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts/evaluations/probability_diagnostics"
)

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


def load_predictions(path: Path, model_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} predictions not found:\n{path}"
        )

    df = pd.read_csv(path)

    required = {
        "StudyInstanceUID",
        "label",
        "actual",
        "probability",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{model_name} file is missing columns: {sorted(missing)}"
        )

    df = df[
        ["StudyInstanceUID", "label", "actual", "probability"]
    ].copy()

    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    df["label"] = df["label"].astype(str)
    df["actual"] = pd.to_numeric(
        df["actual"], errors="raise"
    ).astype(int)
    df["probability"] = pd.to_numeric(
        df["probability"], errors="raise"
    )

    if not df["actual"].isin([0, 1]).all():
        raise ValueError("'actual' must contain only 0 or 1.")

    if not df["probability"].between(0, 1).all():
        raise ValueError("Probabilities must be between 0 and 1.")

    if df.duplicated(["StudyInstanceUID", "label"]).any():
        raise ValueError(
            f"{model_name} contains duplicate study/label rows."
        )

    return df


def summarize_distributions(
    df: pd.DataFrame,
    model_name: str,
) -> list[dict]:
    rows = []

    for label in LABELS:
        label_df = df[df["label"] == label]

        positive = label_df.loc[
            label_df["actual"] == 1, "probability"
        ].to_numpy()

        negative = label_df.loc[
            label_df["actual"] == 0, "probability"
        ].to_numpy()

        pos_mean = float(np.mean(positive)) if len(positive) else np.nan
        neg_mean = float(np.mean(negative)) if len(negative) else np.nan

        rows.append({
            "model": model_name,
            "label": label,
            "positive_count": len(positive),
            "negative_count": len(negative),
            "positive_mean": pos_mean,
            "negative_mean": neg_mean,
            "mean_difference_pos_minus_neg": (
                pos_mean - neg_mean
                if len(positive) and len(negative)
                else np.nan
            ),
            "positive_min": (
                float(np.min(positive)) if len(positive) else np.nan
            ),
            "positive_max": (
                float(np.max(positive)) if len(positive) else np.nan
            ),
            "negative_min": (
                float(np.min(negative)) if len(negative) else np.nan
            ),
            "negative_max": (
                float(np.max(negative)) if len(negative) else np.nan
            ),
        })

    return rows


def plot_distributions(
    baseline: pd.DataFrame,
    weighted: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save one figure per model, with positive and negative cases per label."""

    for model_name, df in [
        ("baseline", baseline),
        ("weighted", weighted),
    ]:
        fig, ax = plt.subplots(figsize=(13, 7))

        positions = np.arange(len(LABELS))
        positive_data = []
        negative_data = []

        for label in LABELS:
            part = df[df["label"] == label]

            positive_data.append(
                part.loc[part["actual"] == 1, "probability"].to_numpy()
            )
            negative_data.append(
                part.loc[part["actual"] == 0, "probability"].to_numpy()
            )

        # Side-by-side box plots show probability spread by actual class.
        offset = 0.18
        ax.boxplot(
            positive_data,
            positions=positions - offset,
            widths=0.28,
            patch_artist=False,
            manage_ticks=False,
        )
        ax.boxplot(
            negative_data,
            positions=positions + offset,
            widths=0.28,
            patch_artist=False,
            manage_ticks=False,
        )

        ax.set_xticks(positions)
        ax.set_xticklabels(LABELS, rotation=40, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Predicted probability")
        ax.set_title(
            f"{model_name.title()} model: probabilities by actual outcome"
        )
        ax.axhline(0.5, linestyle="--", linewidth=1)
        ax.grid(axis="y", alpha=0.3)

        ax.text(
            0.01, 0.98,
            "Left box: actual positive (1)    Right box: actual negative (0)",
            transform=ax.transAxes,
            va="top",
        )

        fig.tight_layout()
        fig.savefig(
            output_dir / f"{model_name}_probability_distributions.png",
            dpi=150,
        )
        plt.close(fig)


def main():
    baseline = load_predictions(BASELINE_FILE, "Baseline")
    weighted = load_predictions(WEIGHTED_FILE, "Weighted")

    baseline = baseline.sort_values(
        ["StudyInstanceUID", "label"]
    ).reset_index(drop=True)

    weighted = weighted.sort_values(
        ["StudyInstanceUID", "label"]
    ).reset_index(drop=True)

    keys = ["StudyInstanceUID", "label"]

    if not baseline[keys].equals(weighted[keys]):
        raise ValueError(
            "Baseline and weighted files do not have identical study/label keys."
        )

    if not np.array_equal(
        baseline["actual"].to_numpy(),
        weighted["actual"].to_numpy(),
    ):
        raise ValueError(
            "Baseline and weighted files have different actual labels."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(summarize_distributions(baseline, "baseline"))
    rows.extend(summarize_distributions(weighted, "weighted"))

    summary = pd.DataFrame(rows)
    summary_path = OUTPUT_DIR / "probability_distribution_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_distributions(baseline, weighted, OUTPUT_DIR)

    print("=" * 72)
    print("PROBABILITY DISTRIBUTION DIAGNOSTIC")
    print("=" * 72)

    print(
        summary[
            [
                "model",
                "label",
                "positive_count",
                "negative_count",
                "positive_mean",
                "negative_mean",
                "mean_difference_pos_minus_neg",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print(f"\nSummary CSV: {summary_path}")
    print(f"Plots saved in: {OUTPUT_DIR}")
    print("\nInterpretation:")
    print("- Positive mean higher than negative mean is encouraging.")
    print("- Similar means suggest weak separation for that label.")
    print("- These means are descriptive; they do not replace AUC or validation.")
    print("- The validation set contains only 17 studies.")


if __name__ == "__main__":
    main()
