from torch.utils.data import DataLoader
from .pytorch_knee_dataset import PyTorchKneeDataset

class ModelDataLoader:

    """
    CReates Mytorch Dataloaders for knee MRI training
    """

    def __init__(
            self,
            dataset: PyTorchKneeDataset,
            batch_size: int=1,
            shuffle: bool = False,
            num_workers: int = 0
        ) -> None:

        if not isinstance(dataset, PyTorchKneeDataset):
            raise TypeError("dataset must be of PyTorchKneeDataset instance")

        if batch_size <= 0:
            raise ValueError("batch size must be positive")

        if num_workers < 0:
            raise ValueError("num_workers cannot be negative")

        self._loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

    def get_loader(self) -> DataLoader:
        return self._loader
