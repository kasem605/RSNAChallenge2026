
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ish_knee.dataset.knee_mri_dataset import KneeMRIDataset
from ish_knee.model.knee_3d_cnn import Knee3DCNN


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


def predict(model, loader):
    """Return study IDs, actual labels, and predicted probabilities."""
    model.eval()
    results = []

    with torch.no_grad():
        for study_uids, images, labels in loader:
            logits = model(
                images[:, 0],
                images[:, 1],
                images[:, 2],
            )
            probabilities = torch.sigmoid(logits)

            for i, study_uid in enumerate(study_uids):
                results.append(
                    (
                        study_uid,
                        labels[i].cpu(),
                        probabilities[i].cpu(),
                    )
                )

    return results


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    manifest = (
        PROJECT_ROOT
        / "outputs"
        / "processed_dataset_v1"
        / "manifest.csv"
    )

    dataset = KneeMRIDataset(
        manifest_path=manifest,
        split="train",
    )

    # Keep the same four training studies.
    subset = Subset(dataset, [0, 1, 2, 3])

    # Fixed order makes before/after comparisons straightforward.
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = Knee3DCNN()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    # Capture predictions before training.
    initial_predictions = predict(model, loader)

    initial_loss = 0.0
    with torch.no_grad():
        for _, images, labels in loader:
            logits = model(
                images[:, 0],
                images[:, 1],
                images[:, 2],
            )
            initial_loss += criterion(logits, labels).item()

    initial_loss /= len(subset)
    print(f"\nInitial four-study loss: {initial_loss:.6f}")

    # Train on the same four studies for 100 epochs.
    for epoch in range(1, 101):
        model.train()
        total_loss = 0.0

        for _, images, labels in loader:
            optimizer.zero_grad(set_to_none=True)

            logits = model(
                images[:, 0],
                images[:, 1],
                images[:, 2],
            )
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        mean_loss = total_loss / len(subset)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}/100 | "
                f"loss={mean_loss:.6f}"
            )

    # Capture predictions after training.
    final_predictions = predict(model, loader)

    print("\nPrediction comparison by study:")
    print("Probability change = final probability - initial probability")

    for before, after in zip(
        initial_predictions,
        final_predictions,
    ):
        study_uid, actual, initial_probs = before
        _, _, final_probs = after

        print(f"\nStudy: {study_uid}")
        print(f"{'Label':<20} {'Actual':>6} {'Before':>9} "
              f"{'After':>9} {'Change':>9}")

        for i, label_name in enumerate(LABEL_NAMES):
            initial_p = initial_probs[i].item()
            final_p = final_probs[i].item()
            change = final_p - initial_p

            print(
                f"{label_name:<20} "
                f"{actual[i].item():>6.0f} "
                f"{initial_p:>9.3f} "
                f"{final_p:>9.3f} "
                f"{change:>+9.3f}"
            )

    # Summarize changes across the four studies for each label.
    print("\nPer-label learning summary:")
    print(
        f"{'Label':<20} {'Positives':>9} {'Before MAE':>11} "
        f"{'After MAE':>10} {'Mean |Change|':>14}"
    )

    for i, label_name in enumerate(LABEL_NAMES):
        actual_values = torch.stack(
            [item[1] for item in final_predictions]
        )[:, i].float()

        initial_values = torch.stack(
            [item[2] for item in initial_predictions]
        )[:, i]

        final_values = torch.stack(
            [item[2] for item in final_predictions]
        )[:, i]

        before_mae = torch.abs(initial_values - actual_values).mean().item()
        after_mae = torch.abs(final_values - actual_values).mean().item()
        mean_change = torch.abs(
            final_values - initial_values
        ).mean().item()

        positives = int(actual_values.sum().item())

        print(
            f"{label_name:<20} "
            f"{positives:>9} "
            f"{before_mae:>11.3f} "
            f"{after_mae:>10.3f} "
            f"{mean_change:>14.3f}"
        )

    print("\nOverfit diagnostic completed.")


if __name__ == "__main__":
    main()
