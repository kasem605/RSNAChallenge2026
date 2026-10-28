"""
Test loading the trained Knee3DCNN checkpoint and running inference

Run from the project root:
"""

from pathlib import Path
import torch
from ish_knee.model.knee_3d_cnn import Knee3DCNN

def main():

    print("=" * 70)
    print("KNEE CNN CHECKPOINT /INFERENCE TEST")
    print("=" * 70)

    checkpoint_path = Path("checkpoints/knee_3d_cnn_baseline_epoch_1.pt")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path.resolve()}")

    # This baseline was trained on CPU
    device = torch.device("cpu")

    # Initialize the same model architecture
    model = Knee3DCNN().to(device)

    # Load the checkpoint
    # weights only=False is used because this checkpoint also contains
    # optimizer state, metadata, and study UID lists
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint loaded: {checkpoint_path.resolve()}")
    print(f"Checkpoint epoch:        {checkpoint['epoch']}")
    print(f"Device:                               {device}")

    # synthetic input verifies model loading and forward inference
    # Each input represents one preprocessed 

    sagittal = torch.rand(1, 64, 64, 64, device=device)
    coronal = torch.rand(1, 64, 64, 64, device=device)
    axial = torch.rand(1, 64, 64, 64, device=device)

    with torch.no_grad():
        logits = model(sagittal, coronal, axial)
        probabilities = torch.sigmoid(logits)

    if probabilities.shape != (1, 12):
        raise RuntimeError(f"Extected prediction shape (1, 12) has {tuple(probabilities.shape)}")

    if not torch.isfinite(probabilities).all():
        raise RuntimeError("Predictions contain NaM or infinity")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise RuntimeError("Prediction probabilities are outside [0, 1]")

    label_names = checkpoint["label_names"]

    print("\nPrediction probabilities")
    print("-" * 45)

    for label, probability in zip(
        label_names,
        probabilities[0].tolist()
    ):
        print(f"{label:20s}: {probability:.4f}")

    print("-" * 45)
    print(f"Output shape: {tuple(probabilities.shape)}")
    print("Probability range: valid")
    print("\nCHECKPOINT / INFERENCE TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    main()
