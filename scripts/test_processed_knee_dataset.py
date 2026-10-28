
from pathlib import Path

import torch

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

dataset = ProcessedKneeDataset(
    manifest_csv=manifest_csv,
    processed_csv=processed_csv,
)

print("=" * 65)
print("PROCESSED KNEE DATASET TEST")
print("=" * 65)

print(f"Number of studies: {len(dataset)}")

sample = dataset[0]

print(f"Study UID: {sample['study_uid']}")
print(f"Image shape: {tuple(sample['images'].shape)}")
print(f"Image dtype: {sample['images'].dtype}")
print(f"Label shape: {tuple(sample['labels'].shape)}")
print(f"Label dtype: {sample['labels'].dtype}")
print(f"Image minimum: {sample['images'].min().item():.4f}")
print(f"Image maximum: {sample['images'].max().item():.4f}")
print(f"Label values: {sample['labels'].tolist()}")

assert len(dataset) == 58
assert sample["images"].shape == (3, 96, 160, 160)
assert sample["images"].dtype == torch.float32
assert sample["labels"].shape == (12,)
assert sample["labels"].dtype == torch.float32

print("-" * 65)
print("PROCESSED KNEE DATASET TEST PASSED")
print("=" * 65)
