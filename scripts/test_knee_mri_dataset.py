
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from torch.utils.data import DataLoader

from ish_knee.dataset.knee_mri_dataset import KneeMRIDataset


def main() -> None:
    manifest_path = (
        PROJECT_ROOT
        / "outputs"
        / "processed_dataset_v1"
        / "manifest.csv"
    )

    print("=" * 65)
    print("KNEE MRI DATASET TEST")
    print("=" * 65)

    train_dataset = KneeMRIDataset(
        manifest_path=manifest_path,
        split="train",
    )

    validation_dataset = KneeMRIDataset(
        manifest_path=manifest_path,
        split="validation",
    )

    assert len(train_dataset) == 41
    assert len(validation_dataset) == 17

    # Test a single training study.
    study_uid, images, labels = train_dataset[0]

    print("\nSingle study:")
    print(f"Images shape: {tuple(images.shape)}")
    print(f"Images dtype: {images.dtype}")
    print(f"Labels shape: {tuple(labels.shape)}")
    print(f"Labels dtype: {labels.dtype}")
    print(f"Image range: {images.min().item():.4f} "
          f"to {images.max().item():.4f}")
    print(f"Labels: {labels.tolist()}")

    assert images.shape == (3, 128, 128, 128)
    assert images.dtype == torch.float32
    assert labels.shape == (12,)
    assert labels.dtype == torch.float32
    assert torch.isfinite(images).all()
    assert torch.all((labels == 0) | (labels == 1))

    # Test loading a small batch. Batch size 1 is a safe
    # starting point for CPU-based 3D MRI training.
    loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    batch_study_uids, batch_images, batch_labels = next(iter(loader))

    print("\nFirst batch:")
    print(f"Images shape: {tuple(batch_images.shape)}")
    print(f"Labels shape: {tuple(batch_labels.shape)}")

    assert batch_images.shape == (1, 3, 128, 128, 128)
    assert batch_labels.shape == (1, 12)

    # Confirm validation samples load correctly too.
    val_study_uid, val_images, val_labels = validation_dataset[0]

    assert val_images.shape == (3, 128, 128, 128)
    assert val_labels.shape == (12,)

    print("\n" + "=" * 65)
    print("KNEE MRI DATASET TEST PASSED")
    print(f"Training studies:   {len(train_dataset)}")
    print(f"Validation studies: {len(validation_dataset)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
