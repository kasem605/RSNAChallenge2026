"""Small-subset overfit diagnostic; does not alter the baseline checkpoint.
Run from project root:
"""
# python -u .\scripts\diagnose_small_subset_overfit.py

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder
from ish_knee.dataset.labeled_dataset_builder import LabeledDatasetBuilder
from ish_knee.dataset.knee_mri_processor import KneeMRIProcessor
from ish_knee.preprocessing.volume.preprocessing_config import PreprocessingConfig
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
from ish_knee.preprocessing.volume.volume_preprocessor import VolumePreprocessor
from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.model.knee_3d_cnn import Knee3DCNN
from ish_knee.model.study_split import StudySplit
from ish_knee.model.model_dataset import ModelDataset
from ish_knee.model.model_sample_builder import ModelSampleBuilder
from ish_knee.model.pytorch_knee_dataset import PyTorchKneeDataset

DATA_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")
VALIDATION_FRACTION = 0.30
RANDOM_SEED = 42
TARGET_SPACING = VoxelSpacing(
    spacing_axis_0=1.0,
    spacing_axis_1=1.0,
    spacing_axis_2=1.0,
)
TARGET_SHAPE = (64, 64, 64)
SUBSET_SIZE = 8
OPTIMIZER_UPDATES = 200
LEARNING_RATE = 0.001
LABEL_NAMES = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "overfit_diagnostic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cpu")

    print("=" * 75)
    print("RSNA KNEE AI - SMALL-SUBSET OVERFIT DIAGNOSTIC")
    print(f"Dataset root: {DATA_ROOT}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Subset: {SUBSET_SIZE} studies; updates: {OPTIMIZER_UPDATES}")
    print("The baseline checkpoint will not be modified.")

    print("\n[1/4] Loading metadata and building fully labeled studies...")
    paths = DatasetPaths.from_root(DATA_ROOT)
    metadata = MetadataReader(paths)
    metadata.validate()
    selector = SeriesSelector(paths.train_series_dir)
    samples = LabeledDatasetBuilder(
        DatasetSampleBuilder(metadata, selector)
    ).build_all()
    train_samples, validation_samples = StudySplit.split(
        samples, validation_fraction=VALIDATION_FRACTION, seed=RANDOM_SEED
    )
    print(f"Fully labeled: {len(samples)}; train: {len(train_samples)}; "
          f"validation: {len(validation_samples)}")

    targets_np = np.asarray(
        [sample.labels.as_tuple for sample in train_samples], dtype=np.float32
    )
    positive_counts = targets_np.sum(axis=0)
    negative_counts = len(train_samples) - positive_counts
    weights = np.clip(
        negative_counts / np.maximum(positive_counts, 1), 1.0, 10.0
    ).astype(np.float32)

    print("\n[2/4] Building the same preprocessing pipeline...")
    config = PreprocessingConfig(
        target_spacing=TARGET_SPACING,
        target_shape=TARGET_SHAPE,
        normalize_intensity=True,
        padding_value=0.0,
    )
    processor = KneeMRIProcessor(VolumePreprocessor(config))
    model_dataset = ModelDataset(train_samples, ModelSampleBuilder(processor))
    torch_dataset = PyTorchKneeDataset(model_dataset)
    n = min(SUBSET_SIZE, len(torch_dataset))
    if n < 2:
        raise ValueError("Fewer than two training studies are available.")

    print(f"\nCaching {n} preprocessed studies once; CPU preprocessing may take time.")
    sagittal_list, coronal_list, axial_list, target_list = [], [], [], []
    for i in range(n):
        sag, cor, ax, target = torch_dataset[i]
        sagittal_list.append(torch.as_tensor(sag).detach().cpu().float())
        coronal_list.append(torch.as_tensor(cor).detach().cpu().float())
        axial_list.append(torch.as_tensor(ax).detach().cpu().float())
        target_list.append(torch.as_tensor(target).detach().cpu().float())
        print(f"  Cached {i + 1}/{n}")

    subset = TensorDataset(
        torch.stack(sagittal_list),
        torch.stack(coronal_list),
        torch.stack(axial_list),
        torch.stack(target_list),
    )
    if subset.tensors[3].ndim != 2 or subset.tensors[3].shape[1] != 12:
        raise ValueError(f"Expected target shape (N, 12); got {subset.tensors[3].shape}")

    print("\nCached tensor shapes:")
    for name, tensor in zip(
        ("sagittal", "coronal", "axial", "targets"), subset.tensors
    ):
        print(f"  {name}: {tuple(tensor.shape)}")

    loader = DataLoader(subset, batch_size=1, shuffle=True, num_workers=0)
    model = Knee3DCNN().to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def evaluate_subset_loss():
        model.eval()
        total = 0.0
        with torch.no_grad():
            for sag, cor, ax, target in DataLoader(subset, batch_size=1, shuffle=False):
                logits = model(sag.float(), cor.float(), ax.float())
                total += criterion(logits, target.float()).item()
        model.train()
        return total / len(subset)

    print("\n[3/4] Running overfit diagnostic...")
    initial_loss = evaluate_subset_loss()
    history = [{"update": 0, "weighted_subset_loss": initial_loss}]
    print(f"Initial weighted subset loss: {initial_loss:.6f}")
    model.train()
    update = 0
    while update < OPTIMIZER_UPDATES:
        for sag, cor, ax, target in loader:
            if update >= OPTIMIZER_UPDATES:
                break
            optimizer.zero_grad(set_to_none=True)
            logits = model(sag.float(), cor.float(), ax.float())
            loss = criterion(logits, target.float())
            loss.backward()
            optimizer.step()
            update += 1
            if update % 20 == 0 or update == OPTIMIZER_UPDATES:
                current = evaluate_subset_loss()
                history.append({"update": update, "weighted_subset_loss": current})
                print(f"Update {update:3d}/{OPTIMIZER_UPDATES} - loss: {current:.6f}")

    model.eval()
    probabilities, target_batches = [], []
    with torch.no_grad():
        for sag, cor, ax, target in DataLoader(subset, batch_size=1, shuffle=False):
            probabilities.append(torch.sigmoid(model(sag.float(), cor.float(), ax.float())))
            target_batches.append(target)
    probs = torch.cat(probabilities, dim=0)
    target_tensor = torch.cat(target_batches, dim=0).int()
    predictions = (probs >= 0.5).int()

    metrics = {}
    print("\nPer-label predictions on these same training studies (threshold 0.5)")
    print(f"{'Label':20s}{'Pos target':>12s}{'Pred pos':>10s}{'TP':>6s}{'FP':>6s}{'FN':>6s}")
    for i, label in enumerate(LABEL_NAMES):
        truth, pred = target_tensor[:, i], predictions[:, i]
        tp = int(((truth == 1) & (pred == 1)).sum())
        fp = int(((truth == 0) & (pred == 1)).sum())
        fn = int(((truth == 1) & (pred == 0)).sum())
        tn = int(((truth == 0) & (pred == 0)).sum())
        metrics[label] = {
            "positive_targets": int(truth.sum()),
            "predicted_positive": int(pred.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
        print(f"{label:20s}{int(truth.sum()):12d}{int(pred.sum()):10d}{tp:6d}{fp:6d}{fn:6d}")

    selected_uids = [s.study_instance_uid for s in train_samples[:n]]
    checkpoint_path = OUTPUT_DIR / "small_subset_overfit_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_updates": OPTIMIZER_UPDATES,
        "selected_training_study_uids": selected_uids,
        "label_names": LABEL_NAMES,
        "positive_class_weights": weights.tolist(),
        "loss_history": history,
        "target_shape": TARGET_SHAPE,
        "target_spacing": TARGET_SPACING.as_tuple,
        "random_seed": RANDOM_SEED,
    }, checkpoint_path)

    results = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "dataset_root": str(DATA_ROOT),
        "subset_size": n,
        "optimizer_updates": OPTIMIZER_UPDATES,
        "learning_rate": LEARNING_RATE,
        "selected_training_study_uids": selected_uids,
        "label_names": LABEL_NAMES,
        "positive_class_weights": weights.tolist(),
        "initial_weighted_subset_loss": initial_loss,
        "final_weighted_subset_loss": history[-1]["weighted_subset_loss"],
        "loss_history": history,
        "per_label_predictions_threshold_0_5": metrics,
        "checkpoint_path": str(checkpoint_path.resolve()),
    }
    results_path = OUTPUT_DIR / "small_subset_overfit_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, allow_nan=False)

    print("\n[4/4] Diagnostic completed.")
    print(f"Results:    {results_path.resolve()}")
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print("This is a training-subset diagnostic, not a validation-performance estimate.")


if __name__ == "__main__":
    main()
