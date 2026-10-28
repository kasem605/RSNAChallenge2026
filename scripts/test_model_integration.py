
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from torch import nn
from torch.utils.data import DataLoader

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
    print("KNEE 3D CNN INTEGRATION TEST")
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

    images, labels = next(iter(loader))

    # Dataset images: (batch, 3, depth, height, width)
    # Model inputs:   (batch, depth, height, width)
    sagittal = images[:, 0]
    coronal = images[:, 1]
    axial = images[:, 2]

    print(f"Sagittal shape: {tuple(sagittal.shape)}")
    print(f"Coronal shape:  {tuple(coronal.shape)}")
    print(f"Axial shape:    {tuple(axial.shape)}")
    print(f"Labels shape:   {tuple(labels.shape)}")

    model = Knee3DCNN()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    optimizer.zero_grad()

    logits = model(sagittal, coronal, axial)

    print(f"Logits shape:   {tuple(logits.shape)}")

    assert logits.shape == (1, 12), (
        f"Expected (1, 12), got {tuple(logits.shape)}"
    )

    loss = criterion(logits, labels)

    if not torch.isfinite(loss):
        raise ValueError(f"Loss is not finite: {loss.item()}")

    print(f"Training loss:  {loss.item():.6f}")

    loss.backward()

    gradients_exist = any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if not gradients_exist:
        raise RuntimeError("No valid model gradients were produced")

    optimizer.step()

    print("\n" + "=" * 65)
    print("MODEL INTEGRATION TEST PASSED")
    print("Three MRI inputs accepted.")
    print("12 output logits produced.")
    print("Loss and backward pass succeeded.")
    print("=" * 65)


if __name__ == "__main__":
    main()
