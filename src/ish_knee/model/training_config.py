from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class TrainingConfig:

    """
    Defines configuration settings to train the
    RSNA Knee Abmormality detection Model
    """

    batch_size: int = 1
    learning_rate: float = 0.001
    epochs: int = 10
    num_workers: int = 0
    checkpoint_dir: Path=Path("checkpoints")

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")

        if self.epochs < 1:
            raise ValueError("num_workers cannot be negative.")

        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

        if not isinstance(self.checkpoint_dir, Path):
            raise TypeError("checkpoint_dir must be a Pathl;[]")

