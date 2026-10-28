
import torch

from ish_knee.model.knee_3d_cnn import Knee3DCNN


def main():

    print("=" * 70)
    print("KNEE 3D CNN TEST")
    print("=" * 70)

    torch.manual_seed(42)

    model = Knee3DCNN()
    model.eval()

    # -----------------------------------------------------------------
    # Simulate one batch using the shapes from ModelDataLoader
    # -----------------------------------------------------------------

    sagittal = torch.randn(1, 64, 64, 64)
    coronal = torch.randn(1, 64, 64, 64)
    axial = torch.randn(1, 64, 64, 64)

    with torch.no_grad():
        logits = model(sagittal, coronal, axial)

    assert tuple(logits.shape) == (1, 12)
    assert torch.isfinite(logits).all()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Sagittal input:", tuple(sagittal.shape))
    print("Coronal input:", tuple(coronal.shape))
    print("Axial input:", tuple(axial.shape))    
    print("Output shape:", tuple(logits.shape))
    print("Paremeters:", f"{parameter_count:,}")

    probabilities = torch.sigmoid(logits)

    assert tuple(probabilities.shape) == (1,12)
    assert torch.all(probabilities >= 0)
    assert torch.all(probabilities <= 1)

    print("Probability shape:", tuple(probabilities.shape))
    print()
    print("KNEE 3D CNN TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    main()