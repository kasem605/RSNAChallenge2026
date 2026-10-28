
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

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    _, images, labels = next(iter(loader))

    model = Knee3DCNN()
    criterion = nn.BCEWithLogitsLoss()

    logits = model(
        images[:, 0],
        images[:, 1],
        images[:, 2],
    )
    loss = criterion(logits, labels)
    loss.backward()

    print(f"\nLoss: {loss.item():.6f}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print("\nGradient summary:")

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            print(f"{name}: NO GRADIENT")
        else:
            grad = parameter.grad
            print(
                f"{name}: "
                f"mean_abs={grad.abs().mean().item():.8e}, "
                f"max_abs={grad.abs().max().item():.8e}, "
                f"finite={torch.isfinite(grad).all().item()}"
            )


if __name__ == "__main__":
    main()
