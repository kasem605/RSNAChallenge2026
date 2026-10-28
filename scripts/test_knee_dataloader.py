
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from ish_knee.dataset.processed_knee_dataset import ProcessedKneeDataset


ROOT = Path(__file__).resolve().parents[1]

manifest_csv = (
    ROOT / "outputs" / "dataset_manifest"
    / "study_series_manifest.csv"
)

processed_csv = (
    ROOT / "outputs" / "processed_labeled_dataset"
    / "processed_volumes.csv"
)

# Load the dataset you already tested.
dataset = ProcessedKneeDataset(
    manifest_csv=manifest_csv,
    processed_csv=processed_csv,
)

# Split by study index, not by individual MRI plane.
indices = list(range(len(dataset)))

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    shuffle=True,
)

# Subsets preserve the original dataset and its labels.
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

print("=" * 65)
print("KNEE DATALOADER TEST")
print("=" * 65)

print(f"Total studies: {len(dataset)}")
print(f"Training studies: {len(train_dataset)}")
print(f"Validation studies: {len(val_dataset)}")

# Confirm no study indices overlap.
assert set(train_indices).isdisjoint(set(val_indices))
assert len(train_indices) + len(val_indices) == len(dataset)

# Inspect one training batch.
batch = next(iter(train_loader))

print(f"Batch images shape: {tuple(batch['images'].shape)}")
print(f"Batch labels shape: {tuple(batch['labels'].shape)}")
print(f"Image dtype: {batch['images'].dtype}")
print(f"Label dtype: {batch['labels'].dtype}")
print(f"Study UID: {batch['study_uid'][0]}")

assert batch["images"].shape == (1, 3, 96, 160, 160)
assert batch["labels"].shape == (1, 12)
assert batch["images"].dtype == torch.float32
assert batch["labels"].dtype == torch.float32

print("-" * 65)
print("KNEE DATALOADER TEST PASSED")
print("=" * 65)
