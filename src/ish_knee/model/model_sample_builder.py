from .model_input import ModelInput
from .model_target import ModelTarget
from .model_sample import ModelSample

from ..dataset.dataset_sample import DatasetSample
from ..dataset.knee_mri_processor import KneeMRIProcessor

class ModelSampleBuilder:

    """
    Builds a complete ModelSample from a DatasetSample

    Converts:
        DatasetSample
            -> KneeMRISample
            -> ModelInput

        KneeLabels
            -> ModelTarget
    """

    def __init__(self, mri_processor: KneeMRIProcessor) -> None:

        if not isinstance(mri_processor, KneeMRIProcessor):
            raise "mri_processor must be a KneeMRIProcessor instance"

        self._mri_processor = mri_processor

    def build(self, sample: DatasetSample) -> ModelSample:
        if not isinstance(sample, DatasetSample):
            raise TypeError("sample must be a DatasetSample instance")

        # --------------------------------------------------------------------
        # Process the MRI volumes
        # --------------------------------------------------------------------

        mri_sample = self._mri_processor.process(sample)

        # --------------------------------------------------------------------
        # Convert MRI data into model input
        # --------------------------------------------------------------------   

        model_input = ModelInput.from_mri_sample(mri_sample)

        # --------------------------------------------------------------------
        # Convert labels into modal target
        # -------------------------------------------------------------------- 

        model_target = ModelTarget.from_labels(sample.labels)

        # --------------------------------------------------------------------
        # Combine input and target
        # -------------------------------------------------------------------- 

        return ModelSample(input=model_input, target=model_target)