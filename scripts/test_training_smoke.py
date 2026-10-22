
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ish_knee.dataset.knee_mri_dataset import KneeMRIDataset
from ish_knee.model.knee_3d_cnn import Knee3DCNN


def main() -> None:
    manifest_path = (
        PROJECT_ROOT
        / "outputs"
        / "processed_dataset_v1"
        / "manifest.csv"
    )

    print("=" * 65)
    print("TRAINING SMOKE TEST")
    print("=" * 65)

    dataset = KneeMRIDataset(
        manifest_path=manifest_path,
        split="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    study_uids, images, labels = next(iter(loader))

    sagittal = images[:, 0]
    coronal = images[:, 1]
    axial = images[:, 2]

    model = Knee3DCNN()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(sagittal, coronal, axial)
    loss = criterion(logits, labels)

    assert logits.shape == (1, 12), logits.shape
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss), loss.item()

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert gradients, "No gradients were produced"
    assert all(torch.isfinite(g).all() for g in gradients), (
        "Non-finite gradients detected"
    )

    optimizer.step()

    print(f"Study UID: {study_uids[0]}")
    print(f"Input shape per plane: {tuple(sagittal.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Initial loss: {loss.item():.6f}")
    print(f"Parameters with gradients: {len(gradients)}")
    print("Optimizer step completed")

    print("\nTRAINING SMOKE TEST PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
