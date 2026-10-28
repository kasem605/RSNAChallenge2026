
import torch

from ish_knee.model.knee_3d_cnn import Knee3DCNN


def main():
    torch.manual_seed(42)

    device = torch.device("cpu")
    model = Knee3DCNN().to(device)
    model.train()

    # Synthetic volumes matching the preprocessed input shape.
    sagittal = torch.randn(1, 64, 64, 64, device=device)
    coronal = torch.randn(1, 64, 64, 64, device=device)
    axial = torch.randn(1, 64, 64, 64, device=device)

    # Synthetic multi-label target.
    targets = torch.randint(
        0, 2, (1, 12), device=device
    ).float()

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )

    optimizer.zero_grad(set_to_none=True)

    logits_before = model(sagittal, coronal, axial)
    loss = criterion(logits_before, targets)
    loss.backward()

    print(f"Initial loss: {loss.item():.6f}")
    print(f"Output shape: {tuple(logits_before.shape)}")
    print("\nGradient diagnostics:")

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            print(f"{name}: NO GRADIENT")
        else:
            grad_mean = parameter.grad.abs().mean().item()
            grad_max = parameter.grad.abs().max().item()

            print(
                f"{name}: "
                f"mean_abs_grad={grad_mean:.8e}, "
                f"max_abs_grad={grad_max:.8e}"
            )

    optimizer.step()

    with torch.no_grad():
        logits_after = model(sagittal, coronal, axial)

    change = (logits_after - logits_before.detach()).abs().mean().item()
    print(f"\nMean absolute logit change after update: {change:.8e}")


if __name__ == "__main__":
    main()
