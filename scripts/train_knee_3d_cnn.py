
from pathlib import Path
import csv
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ish_knee.dataset.knee_mri_dataset import KneeMRIDataset
from ish_knee.model.knee_3d_cnn import Knee3DCNN


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "processed_dataset_v1"
    / "manifest.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training_phase3d_v2"

EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
SEED = 42

NUM_TARGETS = 12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(
    model: Knee3DCNN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    training: bool,
) -> float:
    """Run one training or validation epoch and return mean loss."""

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch_index, (study_uids, images, labels) in enumerate(
    loader, start=1
    ):
        # Dataset images: (B, 3, D, H, W)
        # Model inputs:   (B, D, H, W) per plane.
        sagittal = images[:, 0]
        coronal = images[:, 1]
        axial = images[:, 2]

        if labels.shape[1] != NUM_TARGETS:
            raise ValueError(
                f"Expected {NUM_TARGETS} targets, got {labels.shape[1]}"
            )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(sagittal, coronal, axial)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise ValueError(
                    f"Non-finite loss at batch {batch_index}: "
                    f"{loss.item()}"
                )

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        print(
            f"\r{'Train' if training else 'Valid'} "
            f"{batch_index}/{len(loader)} "
            f"loss={loss.item():.5f}",
            end="",
            flush=True,
        )

    print()

    if total_samples == 0:
        raise ValueError("DataLoader produced no samples")

    return total_loss / total_samples


def save_checkpoint(
    path: Path,
    model: Knee3DCNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    validation_loss: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "num_targets": NUM_TARGETS,
            "seed": SEED,
        },
        path,
    )


def main() -> None:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Processed dataset manifest not found: {MANIFEST_PATH}"
        )

    # Do not overwrite outputs from an earlier run.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    set_seed(SEED)

    print("=" * 70)
    print("KNEE 3D CNN TRAINING")
    print("=" * 70)
    print(f"Device:        CPU")
    print(f"Epochs:        {EPOCHS}")
    print(f"Batch size:    {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Output:        {OUTPUT_DIR}")
    print()

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

    model = Knee3DCNN()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history_path = OUTPUT_DIR / "training_history.csv"
    best_checkpoint_path = OUTPUT_DIR / "knee_3d_cnn_best.pt"
    last_checkpoint_path = OUTPUT_DIR / "knee_3d_cnn_last.pt"

    best_validation_loss = float("inf")

    with history_path.open(
        "w", newline="", encoding="utf-8"
    ) as history_file:
        fieldnames = [
            "epoch",
            "train_loss",
            "validation_loss",
            "best_validation_loss",
        ]
        writer = csv.DictWriter(
            history_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        history_file.flush()

        for epoch in range(1, EPOCHS + 1):
            print(f"\nEpoch {epoch}/{EPOCHS}")
            print("-" * 50)

            train_loss = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                training=True,
            )

            with torch.no_grad():
                validation_loss = run_epoch(
                    model=model,
                    loader=validation_loader,
                    criterion=criterion,
                    optimizer=None,
                    training=False,
                )

            is_best = validation_loss < best_validation_loss

            if is_best:
                best_validation_loss = validation_loss

            # Save a recoverable last checkpoint after every epoch.
            save_checkpoint(
                path=last_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
            )

            # Save a separate checkpoint for the best validation loss.
            if is_best:
                save_checkpoint(
                    path=best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                )

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "best_validation_loss": best_validation_loss,
                }
            )
            history_file.flush()

            print(f"Training loss:          {train_loss:.6f}")
            print(f"Validation loss:        {validation_loss:.6f}")
            print(f"Best validation loss:   {best_validation_loss:.6f}")
            print(f"Best checkpoint updated: {'Yes' if is_best else 'No'}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Last checkpoint: {last_checkpoint_path}")
    print(f"Training history: {history_path}")
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
