
from pathlib import Path
import copy
import random

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
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "weighted_bce_comparison_v2"

SEED = 42
EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = 1e-3

LABEL_NAMES = [
    "acl", "mcl", "medial_meniscus", "lateral_meniscus",
    "medial_oa", "lateral_oa", "pf_oa", "effusion",
    "synovitis", "bakers", "contusion", "fracture",
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0

    for study_uids, images, labels in loader:
        images = images.float()
        labels = labels.float()

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images[:, 0], images[:, 1], images[:, 2])
            loss = loss_fn(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples



def predict(model, loader):
    model.eval()
    all_study_uids = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for study_uids, images, labels in loader:
            all_study_uids.extend(study_uids)

            logits = model(
                images[:, 0].float(),
                images[:, 1].float(),
                images[:, 2].float(),
            )

            all_probs.append(
                torch.sigmoid(logits).cpu().numpy()
            )
            all_labels.append(labels.cpu().numpy())

    probabilities = np.concatenate(all_probs, axis=0)
    actual = np.concatenate(all_labels, axis=0)

    if len(all_study_uids) != len(actual):
        raise ValueError(
            "Study ID count does not match prediction count."
        )

    if len(set(all_study_uids)) != len(all_study_uids):
        raise ValueError("Duplicate study IDs found.")

    return all_study_uids, probabilities, actual


def calculate_metrics(probabilities, actual, experiment):
    rows = []

    for i, name in enumerate(LABEL_NAMES):
        p = probabilities[:, i]
        y = actual[:, i].astype(int)
        pred = (p >= 0.5).astype(int)

        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        rows.append({
            "experiment": experiment,
            "label": name,
            "actual_positive": int(y.sum()),
            "predicted_positive": int(pred.sum()),
            "probability_mean": float(p.mean()),
            "probability_std": float(p.std()),
            "precision_at_0_5": precision,
            "recall_at_0_5": recall,
            "f1_at_0_5": f1,
        })

    return rows


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {OUTPUT_ROOT}. "
            "Choose a new output directory to preserve prior results."
        )

    train_ds = KneeMRIDataset(
        manifest_path=MANIFEST_PATH, split="train"
    )
    val_ds = KneeMRIDataset(
        manifest_path=MANIFEST_PATH, split="validation"
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Estimate positive/negative ratios from training labels only.
    train_labels = np.stack([
        train_ds[i][2].numpy()
        for i in range(len(train_ds))
    ])
    positive_counts = train_labels.sum(axis=0)
    negative_counts = len(train_labels) - positive_counts

    if np.any(positive_counts == 0):
        raise ValueError("At least one label has no positive training examples.")

    pos_weight = torch.tensor(
        negative_counts / positive_counts,
        dtype=torch.float32,
    )

    print("\nPositive-class weights:")
    for name, weight in zip(LABEL_NAMES, pos_weight.tolist()):
        print(f"{name:20s} {weight:.3f}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    history_rows, summary_rows, metric_rows = [], [], []

    experiments = [
        ("unweighted", False),
        ("weighted", True),
    ]

    for experiment, use_weights in experiments:
        seed_everything(SEED)

        run_dir = OUTPUT_ROOT / experiment
        run_dir.mkdir(parents=True, exist_ok=False)

        model = Knee3DCNN()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LEARNING_RATE
        )

        if use_weights:
            loss_fn = torch.nn.BCEWithLogitsLoss(
                pos_weight=pos_weight
            )
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0
        final_train_loss = float("nan")

        print(f"\n{'=' * 65}\nEXPERIMENT: {experiment}\n{'=' * 65}")

        for epoch in range(1, EPOCHS + 1):
            train_loss = run_epoch(
                model, train_loader, loss_fn, optimizer
            )
            val_loss = run_epoch(model, val_loader, loss_fn)

            final_train_loss = train_loss

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Train BCE: {train_loss:.6f} | "
                f"Validation BCE: {val_loss:.6f}"
            )

            history_rows.append({
                "experiment": experiment,
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

        if best_state is None:
            raise RuntimeError("No best checkpoint was captured.")

        model.load_state_dict(best_state)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "experiment": experiment,
                "best_epoch": best_epoch,
                "best_validation_loss": best_val_loss,
                "learning_rate": LEARNING_RATE,
                "seed": SEED,
                "pos_weight": pos_weight if use_weights else None,
            },
            run_dir / "best_model.pt",
        )

        study_uids, probs, actual = predict(model, val_loader)

        prediction_rows = []

        for study_index, study_uid in enumerate(study_uids):
            row = {
                "study_instance_uid": study_uid,
                "split": "validation",
                "dataset_index": study_index,
            }

            for i, name in enumerate(LABEL_NAMES):
                row[f"actual_{name}"] = int(actual[study_index, i])
                row[f"prob_{name}"] = float(probs[study_index, i])

            prediction_rows.append(row)

        pd.DataFrame(prediction_rows).to_csv(
            run_dir / "validation_predictions.csv", index=False
        )

        metric_rows.extend(
            calculate_metrics(probs, actual, experiment)
        )

        summary_rows.append({
            "experiment": experiment,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "final_train_loss": final_train_loss,
        })

    pd.DataFrame(history_rows).to_csv(
        OUTPUT_ROOT / "training_history.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_ROOT / "comparison_summary.csv", index=False
    )
    pd.DataFrame(metric_rows).to_csv(
        OUTPUT_ROOT / "validation_metrics.csv", index=False
    )

    print("\nCOMPARISON SUMMARY")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
