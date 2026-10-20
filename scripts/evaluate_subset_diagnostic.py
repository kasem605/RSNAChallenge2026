
import json
from pathlib import Path

import torch
import numpy as np

from ish_knee.data import DatasetPaths, MetadataReader
from ish_knee.preprocessing import SeriesSelector
from ish_knee.preprocessing.volume.preprocessing_config import PreprocessingConfig
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
from ish_knee.preprocessing.volume.volume_preprocessor import VolumePreprocessor

from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder
from ish_knee.dataset.labeled_dataset_builder import LabeledDatasetBuilder
from ish_knee.dataset.knee_mri_processor import KneeMRIProcessor

from ish_knee.model.model_sample_builder import ModelSampleBuilder
from ish_knee.model.model_dataset import ModelDataset
from ish_knee.model.pytorch_knee_dataset import PyTorchKneeDataset
from ish_knee.model.model_data_loader import ModelDataLoader
from ish_knee.model.knee_3d_cnn import Knee3DCNN
from ish_knee.model.study_split import StudySplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

CHECKPOINT_PATH = CHECKPOINT_DIR / "knee_3d_cnn_subset_100_epochs.pt"
RESULTS_PATH = CHECKPOINT_DIR / "subset_diagnostic_100_epoch_label_metrics.json"

LABEL_NAMES = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

def main():
    print("=" * 70)
    print("SUBSET DIAGNOSTIC: PER-LABEL EVALUATION")
    print("=" * 70)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load the checkpoint safely onto the selected device.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    diagnostic_uids = checkpoint["study_uids"]

    # Rebuild the same dataset and study split.
    paths = DatasetPaths.from_root(DATASET_ROOT)
    metadata = MetadataReader(paths)
    metadata.validate()

    selector = SeriesSelector(paths.train_series_dir)
    sample_builder = DatasetSampleBuilder(metadata, selector)
    labeled_builder = LabeledDatasetBuilder(sample_builder)
    samples = labeled_builder.build_all()

    training_samples, _ = StudySplit.split(
        samples,
        validation_fraction=0.30,
        seed=42,
    )

    samples_by_uid = {
        sample.study_instance_uid: sample
        for sample in training_samples
    }

    missing = [
        uid for uid in diagnostic_uids
        if uid not in samples_by_uid
    ]
    if missing:
        raise RuntimeError(
            f"Diagnostic studies missing from the training split: {missing}"
        )

    subset_samples = [
        samples_by_uid[uid] for uid in diagnostic_uids
    ]

    # Reuse the original preprocessing configuration.
    config = PreprocessingConfig(
        target_spacing=VoxelSpacing(1.0, 1.0, 1.0),
        target_shape=(128, 128, 128),
        normalize_intensity=True,
        padding_value=0.0,
    )

    preprocessor = VolumePreprocessor(config)
    mri_processor = KneeMRIProcessor(preprocessor)
    model_sample_builder = ModelSampleBuilder(mri_processor)

    model_dataset = ModelDataset(
        subset_samples,
        model_sample_builder,
    )
    torch_dataset = PyTorchKneeDataset(model_dataset)

    loader = ModelDataLoader(
        torch_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    ).get_loader()

    # Restore the trained diagnostic model.
    model = Knee3DCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    actual_all = []
    predicted_all = []
    probability_all = []

    with torch.no_grad():
        for sagittal, coronal, axial, targets in loader:
            sagittal = sagittal.to(device, dtype=torch.float32)
            coronal = coronal.to(device, dtype=torch.float32)
            axial = axial.to(device, dtype=torch.float32)

            logits = model(sagittal, coronal, axial)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            actual = targets.cpu().numpy()

            probability_all.append(probabilities[0])
            actual_all.append(actual[0])

    actual_all = np.asarray(actual_all, dtype=np.int32)
    probability_all = np.asarray(probability_all)
    predicted_all = (probability_all >= 0.5).astype(np.int32)

    label_results = []

    print(f"\nStudies evaluated: {len(diagnostic_uids)}")
    print("\nPer-label metrics at threshold 0.5")
    print("-" * 100)
    print(
        f"{'Label':<20} {'Actual+':>8} {'Predicted+':>11} "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'F1':>8}"
    )
    print("-" * 100)

    for i, label in enumerate(LABEL_NAMES):
        y_true = actual_all[:, i]
        y_pred = predicted_all[:, i]

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator else 0.0

        result = {
            "label": label,
            "actual_positive": int(y_true.sum()),
            "actual_negative": int(len(y_true) - y_true.sum()),
            "predicted_positive": int(y_pred.sum()),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "f1": f1,
            "probabilities": [
                float(p) for p in probability_all[:, i]
            ],
        }
        label_results.append(result)

        print(
            f"{label:<20} {result['actual_positive']:>8} "
            f"{result['predicted_positive']:>11} "
            f"{tp:>5} {fp:>5} {fn:>5} {tn:>5} {f1:>8.3f}"
        )

    output = {
        "checkpoint": str(CHECKPOINT_PATH),
        "study_uids": diagnostic_uids,
        "threshold": 0.5,
        "label_metrics": label_results,
        "study_predictions": [
            {
                "study_uid": uid,
                "actual": actual_all[i].tolist(),
                "predicted": predicted_all[i].tolist(),
                "probabilities": probability_all[i].tolist(),
            }
            for i, uid in enumerate(diagnostic_uids)
        ],
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print("-" * 100)
    print(f"Results saved to: {RESULTS_PATH}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
