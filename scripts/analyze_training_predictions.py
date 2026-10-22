
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ish_knee.dataset.knee_mri_dataset import KneeMRIDataset
from ish_knee.model.knee_3d_cnn import Knee3DCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "outputs" / "processed_dataset_v1"
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"

TRAINING_DIR = PROJECT_ROOT / "outputs" / "training_phase3d"
CHECKPOINT_PATH = TRAINING_DIR / "knee_3d_cnn_best.pt"

OUTPUT_CSV = TRAINING_DIR / "training_prediction_diagnostic_v2.csv"

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


def load_model(checkpoint_path: Path) -> Knee3DCNN:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model = Knee3DCNN()
    model.load_state_dict(state_dict)
    model.eval()

    return model


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    if OUTPUT_CSV.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_CSV}\n"
            "Rename the existing file or change OUTPUT_CSV. "
            "The script will not overwrite it."
        )

    dataset = KneeMRIDataset(
        manifest_path=MANIFEST_PATH,
        split="train",
    )

    model = load_model(CHECKPOINT_PATH)

    all_probabilities = []
    all_labels = []
    total_loss = 0.0
    study_results = []

    with torch.no_grad():
        for index in range(len(dataset)):
            study_uid, images, labels = dataset[index]

            # Dataset order: sagittal, coronal, axial.
            sagittal = images[0].unsqueeze(0)
            coronal = images[1].unsqueeze(0)
            axial = images[2].unsqueeze(0)

            targets = labels.unsqueeze(0)

            logits = model(sagittal, coronal, axial)

            loss = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                reduction="mean",
            )

            probabilities = torch.sigmoid(logits)[0].cpu().numpy()
            actual = labels.cpu().numpy()

            total_loss += loss.item()
            all_probabilities.append(probabilities)
            all_labels.append(actual)

            row = {
                "study_instance_uid": study_uid,
                "dataset_index": index,
                "bce_loss": loss.item(),
            }

            for j, label_name in enumerate(LABELS):
                row[f"actual_{label_name}"] = int(actual[j])
                row[f"prob_{label_name}"] = float(probabilities[j])

            study_results.append(row)

    probabilities = np.asarray(all_probabilities)
    actual = np.asarray(all_labels)

    mean_loss = total_loss / len(dataset)

    print("\n" + "=" * 85)
    print("TRAINING PREDICTION DIAGNOSTIC")
    print("=" * 85)

    print(f"Training studies: {len(dataset)}")
    print(f"Mean training BCE: {mean_loss:.6f}")
    print("\nPer-label prediction behavior:")
    print(
        f"{'Label':20s} {'Actual+':>8s} {'Mean P':>9s} "
        f"{'Std P':>9s} {'Min P':>9s} {'Max P':>9s} "
        f"{'Pred+@.5':>10s}"
    )

    summary_rows = []

    for j, label_name in enumerate(LABELS):
        p = probabilities[:, j]
        y = actual[:, j]

        predicted_positive = int((p >= 0.5).sum())
        actual_positive = int(y.sum())

        summary = {
            "label": label_name,
            "actual_positives": actual_positive,
            "actual_negatives": int(len(y) - actual_positive),
            "mean_probability": float(p.mean()),
            "std_probability": float(p.std()),
            "min_probability": float(p.min()),
            "max_probability": float(p.max()),
            "predicted_positives_at_0_5": predicted_positive,
            "mean_absolute_error": float(np.abs(p - y).mean()),
        }

        summary_rows.append(summary)

        print(
            f"{label_name:20s} {actual_positive:8d} "
            f"{p.mean():9.4f} {p.std():9.4f} "
            f"{p.min():9.4f} {p.max():9.4f} "
            f"{predicted_positive:10d}"
        )

    print("\n" + "-" * 85)
    print("Interpretation guide:")
    print("- Very small Std P means predictions barely vary across studies.")
    print("- Mean P far from actual positive prevalence may indicate bias.")
    print("- Mean training BCE should be compared with validation BCE.")
    print("- Training predictions alone do not establish generalization.")

    # Save study-level probabilities without modifying the checkpoint.
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(study_results).to_csv(OUTPUT_CSV, index=False)

    summary_path = OUTPUT_CSV.with_name(
        "training_prediction_summary_v2.csv"
    )

    if summary_path.exists():
        raise FileExistsError(
            f"Summary file already exists: {summary_path}. "
            "Rename it or change the output path before rerunning."
        )

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"\nStudy-level results saved to: {OUTPUT_CSV}")
    print(f"Per-label summary saved to:   {summary_path}")


if __name__ == "__main__":
    main()
