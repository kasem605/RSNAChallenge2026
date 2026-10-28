
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ish_knee.dataset.knee_mri_dataset import KneeMRIDataset
from ish_knee.model.knee_3d_cnn import Knee3DCNN


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "outputs" / "processed_dataset_v1"
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "training_phase3d"
    / "knee_3d_cnn_best.pt"
)

EXPECTED_LABELS = [
    "acl",
    "mcl",
    "medial_meniscus",
    "lateral_meniscus",
    "medial_oa",
    "lateral_oa",
    "pf_oa",
    "effusion",
    "synovitis",
    "bakers",
    "contusion",
    "fracture",
]

PLANES = ["sagittal", "coronal", "axial"]


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    # -----------------------------------------------------
    # 1. Check files
    # -----------------------------------------------------

    section("1. CHECK REQUIRED FILES")

    for path in [MANIFEST_PATH, CHECKPOINT_PATH]:
        print(f"{path}: {'FOUND' if path.exists() else 'MISSING'}")

        if not path.exists():
            raise FileNotFoundError(path)

    # -----------------------------------------------------
    # 2. Check manifest and label columns
    # -----------------------------------------------------

    section("2. CHECK LABEL ORDER")

    manifest = pd.read_csv(MANIFEST_PATH)

    print("Manifest rows:", len(manifest))
    print("Manifest columns:", list(manifest.columns))
    print("\nExpected model label order:")

    for index, label in enumerate(EXPECTED_LABELS):
        print(f"  Output {index:2d}: {label}")

    # Common alternatives for the original CSV target names.
    aliases = {
        "acl": ["acl"],
        "mcl": ["mcl"],
        "medial_meniscus": ["medial_meniscus", "Medial Meniscus"],
        "lateral_meniscus": ["lateral_meniscus", "Lateral Meniscus"],
        "medial_oa": ["medial_oa", "Medial OA"],
        "lateral_oa": ["lateral_oa", "Lateral OA"],
        "pf_oa": ["pf_oa", "PF OA"],
        "effusion": ["effusion", "Effusion"],
        "synovitis": ["synovitis", "Synovitis"],
        "bakers": ["bakers", "Baker's"],
        "contusion": ["contusion", "Contusion"],
        "fracture": ["fracture", "Fracture"],
    }

    for label in EXPECTED_LABELS:
        found = [
            column
            for column in aliases[label]
            if column in manifest.columns
        ]

        if found:
            print(f"  {label:20s} -> manifest column: {found[0]}")
        else:
            print(
                f"  {label:20s} -> no direct label column found "
                "(labels may be stored inside NPZ files)"
            )

    # -----------------------------------------------------
    # 3. Load one training sample
    # -----------------------------------------------------

    section("3. CHECK DATASET TENSORS")

    dataset = KneeMRIDataset(
        manifest_path=MANIFEST_PATH,
        split="train",
    )

    if len(dataset) == 0:
        raise RuntimeError("Training dataset is empty.")

    images, labels = dataset[0]

    print("Training dataset size:", len(dataset))
    print("Image tensor shape:", tuple(images.shape))
    print("Label tensor shape:", tuple(labels.shape))
    print("Image tensor dtype:", images.dtype)
    print("Label tensor dtype:", labels.dtype)

    if tuple(images.shape) != (3, 128, 128, 128):
        raise ValueError(
            f"Unexpected image shape: {tuple(images.shape)}"
        )

    if tuple(labels.shape) != (12,):
        raise ValueError(
            f"Expected 12 labels, got {tuple(labels.shape)}"
        )

    if not torch.isfinite(images).all():
        raise ValueError("Image tensor contains NaN or infinity.")

    if not torch.isfinite(labels).all():
        raise ValueError("Label tensor contains NaN or infinity.")

    unique_labels = torch.unique(labels).tolist()

    if not all(value in (0, 1) for value in unique_labels):
        raise ValueError(
            f"Labels are not binary 0/1 values: {unique_labels}"
        )

    print("Image min/max:", images.min().item(), images.max().item())
    print("Unique label values:", unique_labels)
    print("\nLabels in dataset output order:")

    for index, label_name in enumerate(EXPECTED_LABELS):
        print(f"  {index:2d} {label_name:20s}: {labels[index].item():.0f}")

    # -----------------------------------------------------
    # 4. Check each imaging branch
    # -----------------------------------------------------

    section("4. CHECK THREE MODEL INPUT BRANCHES")

    for index, plane in enumerate(PLANES):
        volume = images[index]

        print(
            f"{plane:8s}: shape={tuple(volume.shape)}, "
            f"min={volume.min().item():.4f}, "
            f"max={volume.max().item():.4f}, "
            f"mean={volume.mean().item():.4f}, "
            f"std={volume.std().item():.4f}"
        )

        if volume.std().item() < 1e-6:
            raise ValueError(f"{plane} volume is nearly constant.")

    # -----------------------------------------------------
    # 5. Load the saved checkpoint
    # -----------------------------------------------------

    section("5. LOAD BEST CHECKPOINT")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(
            checkpoint["model"], dict
        ):
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        raise TypeError(
            f"Unexpected checkpoint type: {type(checkpoint)}"
        )

    model = Knee3DCNN()
    model.load_state_dict(state_dict)
    model.eval()

    print("Checkpoint loaded successfully.")

    # -----------------------------------------------------
    # 6. Run one forward pass
    # -----------------------------------------------------

    section("6. RUN MODEL FORWARD PASS")

    batch = images.unsqueeze(0)

    sagittal = batch[:, 0]
    coronal = batch[:, 1]
    axial = batch[:, 2]

    print("Sagittal input:", tuple(sagittal.shape))
    print("Coronal input: ", tuple(coronal.shape))
    print("Axial input:   ", tuple(axial.shape))

    with torch.no_grad():
        logits = model(sagittal, coronal, axial)
        probabilities = torch.sigmoid(logits)

    print("Logits shape:", tuple(logits.shape))
    print("Probability shape:", tuple(probabilities.shape))

    if tuple(logits.shape) != (1, 12):
        raise ValueError(
            f"Expected model output (1, 12), got {tuple(logits.shape)}"
        )

    if not torch.isfinite(logits).all():
        raise ValueError("Model logits contain NaN or infinity.")

    print("\nOutput probabilities for this training sample:")

    for index, label_name in enumerate(EXPECTED_LABELS):
        print(
            f"  {label_name:20s}: "
            f"{probabilities[0, index].item():.4f}"
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    section("DIAGNOSTIC SUMMARY")

    print("PASS: Dataset sample loaded.")
    print("PASS: Input tensor has expected dimensions.")
    print("PASS: Labels are finite binary values.")
    print("PASS: Three imaging volumes are non-constant.")
    print("PASS: Best checkpoint loaded.")
    print("PASS: Model returned 12 finite logits.")

    print(
        "\nNote: these checks confirm tensor shapes and basic values. "
        "They do not independently prove that the labels stored in "
        "every NPZ file match the intended study or that the "
        "anatomical planes are correctly oriented."
    )


if __name__ == "__main__":
    main()
