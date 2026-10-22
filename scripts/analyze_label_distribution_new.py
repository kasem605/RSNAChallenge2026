
from pathlib import Path
import csv

import numpy as np


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "processed_dataset_v1"
    / "manifest.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "training_phase3d"
    / "label_distribution.csv"
)

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


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}\n"
            "Rename it before running again to preserve prior results."
        )

    with MANIFEST_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("The manifest contains no rows.")

    required_columns = {"split", *LABEL_NAMES}
    missing_columns = required_columns - set(rows[0].keys())

    if missing_columns:
        raise ValueError(
            "Manifest is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    split_names = ["train", "validation"]
    results = []

    print("=" * 95)
    print("TRAINING AND VALIDATION LABEL DISTRIBUTION")
    print("=" * 95)

    for split_name in split_names:
        split_rows = [
            row for row in rows
            if row["split"].strip().lower() == split_name
        ]

        if not split_rows:
            print(f"WARNING: No rows found for split '{split_name}'.")

        print()
        print(f"SPLIT: {split_name.upper()}")
        print(f"Studies: {len(split_rows)}")
        print("-" * 85)
        print(
            f"{'Label':<22}"
            f"{'Positive':>10}"
            f"{'Negative':>10}"
            f"{'Prevalence':>13}"
            f"{'Status':>20}"
        )
        print("-" * 85)

        for label in LABEL_NAMES:
            values = []

            for row in split_rows:
                raw_value = row[label].strip()

                if raw_value not in {"0", "1"}:
                    raise ValueError(
                        f"Unexpected value {raw_value!r} "
                        f"for label '{label}' in split '{split_name}'. "
                        "Expected 0 or 1."
                    )

                values.append(int(raw_value))

            values = np.asarray(values, dtype=np.int64)

            positive = int(np.sum(values == 1))
            negative = int(np.sum(values == 0))
            total = len(values)

            prevalence = positive / total if total else float("nan")

            if positive == 0:
                status = "NO POSITIVES"
            elif negative == 0:
                status = "NO NEGATIVES"
            elif positive < 5:
                status = "FEW POSITIVES"
            else:
                status = "Check counts"

            result = {
                "split": split_name,
                "label": label,
                "study_count": total,
                "positive_count": positive,
                "negative_count": negative,
                "positive_prevalence": (
                    prevalence if total else ""
                ),
                "status": status,
            }

            results.append(result)

            prevalence_text = (
                f"{prevalence:.1%}" if total else "N/A"
            )

            print(
                f"{label:<22}"
                f"{positive:>10}"
                f"{negative:>10}"
                f"{prevalence_text:>13}"
                f"{status:>20}"
            )

        print("-" * 85)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "label",
                "study_count",
                "positive_count",
                "negative_count",
                "positive_prevalence",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print()
    print("Analysis completed.")
    print(f"Results saved to: {OUTPUT_PATH}")
    print("=" * 95)


if __name__ == "__main__":
    main()
