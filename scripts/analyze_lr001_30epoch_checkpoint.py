
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ish_knee.dataset.knee_mri_dataset import KneeMRIDataset
from ish_knee.model.knee_3d_cnn import Knee3DCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_ROOT / "outputs" / "processed_dataset_v1" / "manifest.csv"
)

# Adjust this path if your saved checkpoint is in a different folder.
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "training_lr001_30epochs"
    / "lr_0.001"
    / "best_model.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "lr001_checkpoint_diagnostics"
)

BATCH_SIZE = 1

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


def evaluate(model, loader, split):
    model.eval()
    probability_batches = []
    label_batches = []
    total_loss = 0.0
    count = 0
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

            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels, reduction="sum"
            )

            probabilities = torch.sigmoid(logits)

            total_loss += loss.item()
            count += labels.shape[0]

            probability_batches.append(probabilities.cpu().numpy())
            label_batches.append(labels.cpu().numpy())

    probabilities = np.concatenate(probability_batches, axis=0)
    actual = np.concatenate(label_batches, axis=0)

    print(f"\n{split.upper()} BCE: {total_loss / (count * len(LABEL_NAMES)):.6f}")

    rows = []

    for i, name in enumerate(LABEL_NAMES):
        p = probabilities[:, i]
        y = actual[:, i]

        rows.append({
            "split": split,
            "label": name,
            "actual_positive": int(y.sum()),
            "probability_mean": float(p.mean()),
            "probability_std": float(p.std()),
            "probability_min": float(p.min()),
            "probability_max": float(p.max()),
            "predicted_positive_at_0_5": int((p >= 0.5).sum()),
        })

    return study_uids_all, probabilities, actual, rows


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # Do not overwrite previous diagnostic results.
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {OUTPUT_DIR}"
        )

    device = torch.device("cpu")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model = Knee3DCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Best epoch: {checkpoint['best_epoch']}")
    print(
        "Saved best validation BCE: "
        f"{checkpoint['best_validation_bce']:.6f}"
    )

    all_rows = []

    for split in ["train", "validation"]:
        dataset = KneeMRIDataset(
            manifest_path=MANIFEST_PATH,
            split=split,
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        probabilities, actual, rows = evaluate(
            model, loader, split
        )
        all_rows.extend(rows)

        prediction_rows = []

        for study_index in range(len(actual)):
            row = {"dataset_index": study_index}

            for i, name in enumerate(LABEL_NAMES):
                row[f"actual_{name}"] = int(actual[study_index, i])
                row[f"prob_{name}"] = float(
                    probabilities[study_index, i]
                )

            prediction_rows.append(row)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(prediction_rows).to_csv(
            OUTPUT_DIR / f"{split}_predictions.csv",
            index=False,
        )

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(OUTPUT_DIR / "prediction_diagnostics.csv", index=False)

    print("\nPREDICTION DIAGNOSTICS")
    print(metrics.to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
