"""
Run inference on one real RSNA knee MRI study.
"""
# Run from the project root:
#    python .\scripts\predict_knee_study.py

# Optional:
#    python .\scripts\predict_knee_study.py STUDY_INSTANCE_UID

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

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

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATA_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")

CHECKPOINT_PATH = Path(
    "checkpoints/knee_3d_cnn_baseline_epoch_1.pt"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "predictions"

TARGET_SPACING = VoxelSpacing(
    spacing_axis_0=1.0,
    spacing_axis_1=1.0,
    spacing_axis_2=1.0,
)

TARGET_SHAPE = (64, 64, 64)

# Keep this in the same order used when training.
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
    print("=" * 75)
    print("RSNA KNEE AI - REAL STUDY INFERENCE")
    print("=" * 75)

    device = torch.device("cpu")

    # --------------------------------------------------------
    # 1. Verify required files.
    # --------------------------------------------------------

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH.resolve()}"
        )

    # --------------------------------------------------------
    # 2. Load checkpoint and model.
    # --------------------------------------------------------

    print("\n[1/5] Loading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    checkpoint_labels = checkpoint.get("label_names")

    if checkpoint_labels != LABEL_NAMES:
        raise ValueError(
            "The checkpoint label names or order do not match "
            "the inference script."
        )

    model = Knee3DCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint: {CHECKPOINT_PATH.resolve()}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Device: {device}")

    # --------------------------------------------------------
    # 3. Load metadata and select a complete study.
    # --------------------------------------------------------

    print("\n[2/5] Loading dataset metadata...")

    paths = DatasetPaths.from_root(DATA_ROOT)
    metadata = MetadataReader(paths)
    metadata.validate()

    series_selector = SeriesSelector(paths.train_series_dir)

    sample_builder = DatasetSampleBuilder(
        metadata,
        series_selector,
    )

    labeled_builder = LabeledDatasetBuilder(sample_builder)
    samples = labeled_builder.build_all()

    if not samples:
        raise ValueError(
            "No fully labeled studies were found."
        )

    # An optional study UID can be supplied as the first argument.
    requested_uid = sys.argv[1] if len(sys.argv) > 1 else None

    if requested_uid:
        matching_samples = [
            sample
            for sample in samples
            if sample.study_instance_uid == requested_uid
        ]

        if not matching_samples:
            raise ValueError(
                "The requested study UID was not found among "
                "the fully labeled studies."
            )

        sample = matching_samples[0]

    else:
        # Use a study from the existing validation split, if present.
        # This permits comparison against its known labels for debugging.
        validation_uids = set(
            checkpoint.get("validation_study_uids", [])
        )

        matching_samples = [
            item
            for item in samples
            if item.study_instance_uid in validation_uids
        ]

        if not matching_samples:
            raise ValueError(
                "No checkpoint validation study could be matched "
                "to the currently available labeled studies."
            )

        sample = matching_samples[0]

    study_uid = sample.study_instance_uid

    print(f"Selected study UID: {study_uid}")

    # --------------------------------------------------------
    # 4. Reuse the training preprocessing pipeline.
    # --------------------------------------------------------

    print("\n[3/5] Preprocessing real MRI volumes...")

    preprocessing_config = PreprocessingConfig(
        target_spacing=TARGET_SPACING,
        target_shape=TARGET_SHAPE,
        normalize_intensity=True,
        padding_value=0.0,
    )

    volume_preprocessor = VolumePreprocessor(
        preprocessing_config
    )

    mri_processor = KneeMRIProcessor(volume_preprocessor)
    model_sample_builder = ModelSampleBuilder(mri_processor)

    model_dataset = ModelDataset(
        [sample],
        model_sample_builder,
    )

    model_sample = model_dataset.get_model_sample(0)
    model_input = model_sample.input

    # Each volume should have shape (depth, height, width).
    sagittal = torch.as_tensor(
        model_input.sagittal,
        dtype=torch.float32,
    ).unsqueeze(0)

    coronal = torch.as_tensor(
        model_input.coronal,
        dtype=torch.float32,
    ).unsqueeze(0)

    axial = torch.as_tensor(
        model_input.axial,
        dtype=torch.float32,
    ).unsqueeze(0)

    expected_shape = (1, 64, 64, 64)

    for name, tensor in [
        ("sagittal", sagittal),
        ("coronal", coronal),
        ("axial", axial),
    ]:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} has shape {tuple(tensor.shape)}; "
                f"expected {expected_shape}."
            )

        if not torch.isfinite(tensor).all():
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

    sagittal = sagittal.to(device)
    coronal = coronal.to(device)
    axial = axial.to(device)

    # --------------------------------------------------------
    # 5. Predict and save results.
    # --------------------------------------------------------

    print("\n[4/5] Running inference...")

    with torch.no_grad():
        logits = model(sagittal, coronal, axial)
        probabilities = torch.sigmoid(logits)[0]
        binary_predictions = (probabilities >= 0.5).to(torch.int32)

    if probabilities.shape != (12,):
        raise RuntimeError(
            f"Expected 12 predictions, got {tuple(probabilities.shape)}."
        )

    if not torch.isfinite(probabilities).all():
        raise RuntimeError(
            "The model produced non-finite prediction probabilities."
        )

    predictions = {}

    print("\nPrediction results")
    print("-" * 75)

    for index, label in enumerate(LABEL_NAMES):
        probability = float(probabilities[index].item())
        predicted_label = int(binary_predictions[index].item())

        predictions[label] = {
            "probability": round(probability, 6),
            "predicted_positive": predicted_label,
        }

        print(
            f"{label:20s} "
            f"probability={probability:.4f} "
            f"predicted={predicted_label}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    safe_uid = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in study_uid
    )

    output_path = OUTPUT_DIR / f"{safe_uid}_predictions.json"

    results = {
        "run_timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "study_instance_uid": study_uid,
        "checkpoint_path": str(CHECKPOINT_PATH.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "device": str(device),
        "threshold": 0.5,
        "input_shape_per_plane": list(expected_shape),
        "predictions": predictions,
        "note": (
            "Exploratory baseline predictions; not validated "
            "for clinical use."
        ),
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, allow_nan=False)

    print("\n[5/5] Inference completed.")
    print(f"Results saved to: {output_path.resolve()}")
    print("\nREAL STUDY INFERENCE TEST PASSED")
    print("=" * 75)


if __name__ == "__main__":
    main()