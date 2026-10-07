import torch
from torch.utils.data import Dataset

from .model_dataset import ModelDataset

class PyTorchKneeDataset(Dataset):

    """
    Pytorch adapter for the knee MRI model dataset

    MRI volumes are loaded and converted to tensors on demand
    """

    def __init__(self, model_dataset: ModelDataset) -> None:
        if not isinstance(model_dataset, ModelDataset):
            raise TypeError("model_dataset must be a ModelDataset instance")

        self._model_dataset = model_dataset

    def __len__(self) -> int:
        return len(self._model_dataset)

    def __getitem__(self, index: int):
        model_sample = self._model_dataset.get_model_sample(index)

        sagittal = torch.from_numpy(model_sample.input.sagittal).float()

        coronal = torch.from_numpy(model_sample.input.coronal).float()

        axial = torch.from_numpy(model_sample.input.axial).float()

        target = torch.from_numpy(model_sample.target.values).float()

        return(
            sagittal,
            coronal,
            axial,
            target
        )     
