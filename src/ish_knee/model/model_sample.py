from dataclasses import dataclass

from .model_input import ModelInput
from .model_target import ModelTarget

@dataclass(frozen=True)
class ModelSample:

    """
    Represents one complete model traiing sampl;e

    Contains the processed MRI volumes and their
    corresponding abnormality targets
    """

    input: ModelInput
    target: ModelTarget

    def __post_init__(self) -> None:

        if not isinstance(self.input, ModelInput):
            raise TypeError("input must be a ModelInput instance.")

        if not isinstance(self.target, ModelTarget):
            raise TypeError("input must be a MOdelTarget instance.")       

    @property
    def study_instance_uid(self) -> str:
        return self.input.study_instance_uid