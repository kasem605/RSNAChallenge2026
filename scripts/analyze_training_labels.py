
from pathlib import Path
import csv

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "processed_dataset_v1"
    / "manifest.csv"
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
    with MANIFEST_PATH.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    rows = [row for row in rows if row["split"] == "train"]

    labels = []

    for row in rows:
        file_path = (
            MANIFEST_PATH.parent / row["file_path"]
        ).resolve()

        with np.load(file_path, allow_pickle=False) as data:
            labels.append(data["labels"].copy())

    labels = np.asarray(labels)

    print(f"Training studies: {len(labels)}")
    print(f"Label matrix shape: {labels.shape}")
    print()
    print(
        f"{'Target':<22}"
        f"{'Positive':>10}"
        f"{'Negative':>10}"
        f"{'Positive %':>12}"
    )
    print("-" * 54)

    for index, name in enumerate(LABEL_NAMES):
        positive = int((labels[:, index] == 1).sum())
        negative = int((labels[:, index] == 0).sum())
        total = positive + negative
        percentage = 100.0 * positive / total if total else 0.0

        print(
            f"{name:<22}"
            f"{positive:>10}"
            f"{negative:>10}"
            f"{percentage:>11.1f}%"
        )


if __name__ == "__main__":
    main()
