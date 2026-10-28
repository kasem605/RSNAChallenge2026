from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


# ------------------------------------------------------------
# Project paths and imports
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.preprocessing.dicom.dicom_reader import DicomSeriesReader
from ish_knee.preprocessing.dicom.dicom_volume_builder import DicomVolumeBuilder
from ish_knee.preprocessing.volume.spacing_analyzer import SpacingAnalyzer
from ish_knee.preprocessing.volume.preprocessing_config import (
    PreprocessingConfig,
)
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
from ish_knee.preprocessing.volume.volume_preprocessor import (
    VolumePreprocessor,
)
from ish_knee.model.knee_3d_cnn import Knee3DCNN


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATA_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "training_phase3d_v2"
    / "knee_3d_cnn_best.pt"
)

TARGET_SHAPE = (128, 128, 128)

PLANES = ("sagittal", "coronal", "axial")

EXPECTED_LABELS = [
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


def main() -> None:
    print("=" * 72)
    print("RSNA KNEE - LOCAL TEST INFERENCE")
    print("=" * 72)

    # 1. Validate required files.
    required_files = [
        DATA_ROOT / "test.csv",
        DATA_ROOT / "test_series.csv",
        DATA_ROOT / "sample_submission.csv",
        CHECKPOINT_PATH,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    test_series_dir = DATA_ROOT / "test_series"

    if not test_series_dir.is_dir():
        raise FileNotFoundError(
            f"Test DICOM directory not found: {test_series_dir}"
        )

    # 2. Read test metadata and submission template.
    test_df = pd.read_csv(DATA_ROOT / "test.csv")
    series_df = pd.read_csv(DATA_ROOT / "test_series.csv")
    submission = pd.read_csv(DATA_ROOT / "sample_submission.csv")

    if "StudyInstanceUID" not in test_df.columns:
        raise ValueError("test.csv has no StudyInstanceUID column")

    expected_columns = ["StudyInstanceUID"] + EXPECTED_LABELS

    if submission.columns.tolist() != expected_columns:
        raise ValueError(
            "Unexpected sample submission columns.\n"
            f"Expected: {expected_columns}\n"
            f"Found:    {submission.columns.tolist()}"
        )

    test_uids = test_df["StudyInstanceUID"].astype(str).tolist()
    submission_uids = submission["StudyInstanceUID"].astype(str).tolist()

    if set(test_uids) != set(submission_uids):
        raise ValueError(
            "The study IDs in test.csv do not match sample_submission.csv"
        )

    # 3. Configure the same preprocessing used for training.
    config = PreprocessingConfig(
        target_spacing=VoxelSpacing(1.0, 1.0, 1.0),
        target_shape=TARGET_SHAPE,
        normalize_intensity=True,
        padding_value=0.0,
    )

    preprocessor = VolumePreprocessor(config)
    series_selector = SeriesSelector(test_series_dir)
    series_reader = DicomSeriesReader()
    volume_builder = DicomVolumeBuilder()
    spacing_analyzer = SpacingAnalyzer()

    # 4. Load the trained model.
    device = torch.device("cpu")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a dictionary")

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint has no model_state_dict")

    if checkpoint.get("num_targets", len(EXPECTED_LABELS)) != len(
        EXPECTED_LABELS
    ):
        raise ValueError("Checkpoint target count is not 12")

    model = Knee3DCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Device: {device}")
    print(f"Test studies: {len(test_uids)}")

    # 5. Predict every test study.
    predictions = {}

    for index, study_uid in enumerate(test_uids, start=1):
        print(f"\n[{index}/{len(test_uids)}] Study: {study_uid}")

        selected = series_selector.select(
            study_uid=study_uid,
            series=series_df,
        )

        selected_by_plane = {
            "sagittal": selected.sagittal,
            "coronal": selected.coronal,
            "axial": selected.axial,
        }

        missing_planes = [
            plane
            for plane, series in selected_by_plane.items()
            if series is None
        ]

        if missing_planes:
            raise ValueError(
                f"Study {study_uid} is missing required planes: "
                f"{missing_planes}"
            )

        volumes = []

        for plane in PLANES:
            selected_series = selected_by_plane[plane]
            series_path = Path(selected_series.series_path)

            if not series_path.is_dir():
                raise FileNotFoundError(
                    f"{plane} series directory not found: {series_path}"
                )

            print(
                f"  {plane}: {selected_series.file_count} files"
            )

            dicom_series = series_reader.read(series_path)
            dicom_volume = volume_builder.build(dicom_series)
            current_spacing = spacing_analyzer.analyze(dicom_series)

            processed = preprocessor.preprocess(
                volume=dicom_volume,
                current_spacing=current_spacing,
            )

            volume = np.asarray(
                processed.volume,
                dtype=np.float32,
            )

            if volume.shape != TARGET_SHAPE:
                raise ValueError(
                    f"{study_uid} {plane}: expected {TARGET_SHAPE}, "
                    f"received {volume.shape}"
                )

            if not np.isfinite(volume).all():
                raise ValueError(
                    f"{study_uid} {plane}: volume contains "
                    "NaN or infinity"
                )

            volumes.append(np.ascontiguousarray(volume))

        # Each model input has shape (batch, depth, height, width).
        inputs = [
            torch.from_numpy(volume).unsqueeze(0).to(device)
            for volume in volumes
        ]

        with torch.inference_mode():
            logits = model(
                sagittal=inputs[0],
                coronal=inputs[1],
                axial=inputs[2],
            )

            probabilities = torch.sigmoid(logits).squeeze(0)

        probability_values = probabilities.cpu().numpy()

        if probability_values.shape != (len(EXPECTED_LABELS),):
            raise ValueError(
                f"Unexpected model output shape: "
                f"{probability_values.shape}"
            )

        if not np.isfinite(probability_values).all():
            raise ValueError(
                f"Model returned invalid probabilities for {study_uid}"
            )

        predictions[study_uid] = probability_values

        print(
            "  Prediction range: "
            f"{probability_values.min():.4f} to "
            f"{probability_values.max():.4f}"
        )

    # 6. Fill the template in its original study and column order.
    for row_index, study_uid in enumerate(submission_uids):
        submission.loc[
            row_index, EXPECTED_LABELS
        ] = predictions[study_uid]

    # 7. Save to a unique directory; do not overwrite existing output.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "outputs" / f"test_inference_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    output_csv = output_dir / "submission.csv"
    submission.to_csv(output_csv, index=False)

    # 8. Re-read and validate the saved CSV.
    saved = pd.read_csv(output_csv)

    if saved.columns.tolist() != expected_columns:
        raise ValueError("Saved CSV columns do not match the template")

    if len(saved) != len(test_uids):
        raise ValueError("Saved CSV has an unexpected number of rows")

    numeric_values = saved[EXPECTED_LABELS].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError("Saved CSV contains invalid prediction values")

    if ((numeric_values < 0) | (numeric_values > 1)).any():
        raise ValueError("Predictions must be between 0 and 1")

    print("\n" + "=" * 72)
    print("LOCAL INFERENCE COMPLETED")
    print("=" * 72)
    print(f"Studies predicted: {len(saved)}")
    print(f"Output CSV: {output_csv}")
    print("\nFirst rows:")
    print(saved.head().to_string(index=False))


if __name__ == "__main__":
    main()
