
import torch
from torch import nn

from ish_knee.model.knee_3d_cnn import Knee3DCNN


def main() -> None:
    torch.manual_seed(42)

    device = torch.device("cpu")
    model = Knee3DCNN().to(device)

    # Small synthetic volumes to verify the architecture quickly.
    batch_size = 1
    shape = (32, 32, 32)

    sagittal = torch.randn(
        batch_size, *shape, device=device
    )
    coronal = torch.randn(
        batch_size, *shape, device=device
    )
    axial = torch.randn(
        batch_size, *shape, device=device
    )

    targets = torch.randint(
        0, 2, (batch_size, 12), device=device
    ).float()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(sagittal, coronal, axial)

    assert logits.shape == (batch_size, 12), (
        f"Unexpected output shape: {tuple(logits.shape)}"
    )
    assert torch.isfinite(logits).all(), (
        "Model produced non-finite logits."
    )

    loss = criterion(logits, targets)

    assert torch.isfinite(loss), (
        "Loss is not finite."
    )

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert gradients, "No gradients were produced."
    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
    ), "Non-finite gradients detected."

    optimizer.step()

    print("MODEL SMOKE TEST PASSED")
    print(f"Device:       {device}")
    print(f"Input shape:  {tuple(sagittal.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Initial loss: {loss.item():.6f}")
    print("Backward pass: successful")
    print("Optimizer step: successful")


if __name__ == "__main__":
    main()
