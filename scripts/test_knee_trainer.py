import torch
from torch.utils.data import DataLoader, TensorDataset

from ish_knee.model.knee_3d_cnn import Knee3DCNN
from ish_knee.model.knee_trainer import KneeTrainer
from ish_knee.model.training_config import TrainingConfig

def main() -> None:

    print("=" * 70)
    print("KNEE TRAINER TEST")
    print("=" * 70)

    torch.manual_seed(42)

    # --------------------------------------------------------------
    # Synthetic data verifies tarining mechanics without
    # repeatedly processing large DICOM files
    # --------------------------------------------------------------

    sample_count = 2

    sagittal = torch.randn(sample_count, 64, 64, 64)
    coronal = torch.randn(sample_count, 64, 64, 64)
    axial = torch.randn(sample_count, 64, 64, 64)
    targets = torch.randn(0, 2, (sample_count, 12)).float()

    dataset = TensorDataset(sagittal, coronal, axial, targets)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = Knee3DCNN()
    config = TrainingConfig(batch_size=1, learning_rate=0.001, epochs=1)
    trainer = KneeTrainer(model, config, device=torch.device("cpu"))

    initial_parameters = [
        parameter.detach().clone
        for parameter in model.parameters()
    ]

    average_loss = trainer.train_epoch(loader)

    assert torch.isfinite(torch.tensor(average_loss))
    assert average_loss > 0

    parameters_chamged = any(
        not torch.equal(before,after.detach())
        for before, after in zip(initial_parameters, model.parameters())
    )

    assert parameters_chamged, "Model parameters were not updated"

    print("device:", trainer.device)

    print("Average training loss:", round(average_loss, 6))
    print("Model parametrs updated:", parameters_chamged)
    print()
    print("KNEE TRAINER TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    main()