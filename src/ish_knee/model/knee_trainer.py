import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .knee_3d_cnn import Knee3DCNN
from .training_config import TrainingConfig

class KneeTrainer:

    """
    Trains the KNee3DCNN model using fully labeled MRI studies.
    """

    def __init__(
            self,
            model: Knee3DCNN,
            config: TrainingConfig,
            device: torch.device | None = None
    ) -> None:

        if not isinstance(model, Knee3DCNN):
            raise TypeError("model must be an instance of Knee3DCNN")

        if not isinstance(config, TrainingConfig):
            raise TypeError("config must be a TrainingConfig instance")

        self.model = model

        self.config = config

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)

    def train_epoch(self, loader: DataLoader) -> float:

        """
        Train for one epoch and return the average batch loss.
        """

        self.model.train()

        total_loss = 0.0

        batch_count = 0

        for sagittal, coronal, axial, targets in loader:
            sagittal = sagittal.to(self.device, dtype=torch.float32)
            coronal = coronal.to(self.device, dtype=torch.float32)
            axial = axial.to(self.device, dtype=torch.float32)
            targets = targets.to(self.device, dtype=torch.float32)

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(sagittal, coronal, axial)
            loss = self.criterion(logits, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        if batch_count == 0:
            raise ValueError("The training Dataloader contains no batches")

        return total_loss / batch_count
             

         