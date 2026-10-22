
from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ish_knee.dataset.knee_mri_dataset import KneeMRIDataset
from ish_knee.model.knee_3d_cnn import Knee3DCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_ROOT / "outputs" / "processed_dataset_v1" / "manifest.csv"
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "lr_comparison_v2"

SEED = 42
EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATES = [1e-4, 1e-3]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    sample_count = 0

    for study_uids, images, labels in loader:
        images = images.float()
        labels = labels.float()

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(
                images[:, 0],
                images[:, 1],
                images[:, 2],
            )

            loss = F.binary_cross_entropy_with_logits(
                logits, labels
            )

            if training:
                loss.backward()
                optimizer.step()

        batch_count = images.shape[0]
        total_loss += loss.item() * batch_count
        sample_count += batch_count

    return total_loss / sample_count



def evaluate_predictions(model, loader):
    model.eval()

    all_study_uids = []
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for study_uids, images, labels in loader:
            all_study_uids.extend(study_uids)

            logits = model(
                images[:, 0].float(),
                images[:, 1].float(),
                images[:, 2].float(),
            )

            all_probabilities.append(
                torch.sigmoid(logits).cpu().numpy()
            )
            all_labels.append(labels.cpu().numpy())

    probabilities = np.concatenate(all_probabilities, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    if len(all_study_uids) != len(labels):
        raise ValueError(
            "Study ID count does not match prediction count."
        )

    if len(set(all_study_uids)) != len(all_study_uids):
        raise ValueError(
            "Duplicate study IDs found in validation predictions."
        )

    return all_study_uids, probabilities, labels



def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {OUTPUT_ROOT}\n"
            "Rename it or choose a new output directory. "
            "Existing results will not be overwritten."
        )

    train_dataset = KneeMRIDataset(
        manifest_path=MANIFEST_PATH,
        split="train",
    )
    validation_dataset = KneeMRIDataset(
        manifest_path=MANIFEST_PATH,
        split="validation",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    history_rows = []
    summary_rows = []


    for learning_rate in LEARNING_RATES:
        seed_everything(SEED)

        run_name = f"lr_{learning_rate:g}"
        run_dir = OUTPUT_ROOT / run_name
        run_dir.mkdir(parents=True, exist_ok=False)

        print("\n" + "=" * 70)
        print(f"EXPERIMENT: {run_name}")
        print(f"Learning rate: {learning_rate}")
        print("=" * 70)

        model = Knee3DCNN()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

        best_validation_loss = float("inf")
        best_state = None
        best_epoch = 0

        final_train_loss = None

        for epoch in range(1, EPOCHS + 1):
            train_loss = run_epoch(
                model, train_loader, optimizer
            )

            final_train_loss = train_loss

            validation_loss = run_epoch(
                model, validation_loader
            )

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Train BCE: {train_loss:.6f} | "
                f"Validation BCE: {validation_loss:.6f}"
            )

            history_rows.append({
                "experiment": run_name,
                "learning_rate": learning_rate,
                "epoch": epoch,
                "train_bce": train_loss,
                "validation_bce": validation_loss,
            })

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

        if best_state is None:
            raise RuntimeError("No best model state was captured.")

        model.load_state_dict(best_state)

        checkpoint_path = run_dir / "best_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "learning_rate": learning_rate,
                "best_epoch": best_epoch,
                "best_validation_bce": best_validation_loss,
                "seed": SEED,
            },
            checkpoint_path,
        )

        study_uids, probabilities, actual = evaluate_predictions(
            model, validation_loader
        )

        probability_rows = []

        for study_index, study_uid in enumerate(study_uids):
            row = {
                "study_instance_uid": study_uid,
                "split": "validation",
                "dataset_index": study_index,
            }

            for label_index in range(actual.shape[1]):
                row[f"actual_{label_index}"] = int(
                    actual[study_index, label_index]
                )
                row[f"prob_{label_index}"] = float(
                    probabilities[study_index, label_index]
                )

            probability_rows.append(row)

        pd.DataFrame(probability_rows).to_csv(
            run_dir / "validation_predictions.csv",
            index=False,
        )

        summary_rows.append({
            "experiment": run_name,
            "learning_rate": learning_rate,
            "best_epoch": best_epoch,
            "best_validation_bce": best_validation_loss,
            "final_train_bce": history_rows[-1]["train_bce"],
            "final_train_bce": final_train_loss,
        })

    pd.DataFrame(history_rows).to_csv(
        OUTPUT_ROOT / "learning_rate_history.csv",
        index=False,
    )
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_ROOT / "learning_rate_summary.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
