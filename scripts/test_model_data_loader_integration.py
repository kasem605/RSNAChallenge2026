from pathlib import Path

import torch

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader

from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder
from ish_knee.dataset.labeled_dataset_builder import LabeledDatasetBuilder
from ish_knee.dataset.knee_mri_processor import KneeMRIProcessor

from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.preprocessing.volume.preprocessing_config import PreprocessingConfig
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
from ish_knee.preprocessing.volume.volume_preprocessor import VolumePreprocessor



from ish_knee.model.model_sample_builder import ModelSampleBuilder
from ish_knee.model.model_dataset import ModelDataset
from ish_knee.model.pytorch_knee_dataset import PyTorchKneeDataset
from ish_knee.model.model_data_loader import MOdelDataLoader

def main():

    print("=" * 70)
    print("MODEL DATA LOADER INTEGRATION TEST")
    print("=" * 70)

    # ----------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------

    dataset_root = Path(
        r"D:\RSNA knee abnormality detection 2026"
    )

    paths = DatasetPaths.from_root(dataset_root)

    metadata = MetadataReader(paths)
    metadata.validate()

    series_selector = SeriesSelector(paths.train_series_dir) 

    dataset_sample_builder = DatasetSampleBuilder(metadata, series_selector)

    labeled_builder = LabeledDatasetBuilder(dataset_sample_builder)
    samples=labeled_builder.build_all()

    assert len(samples) == 58
    print("Labeled studies:", len(samples))

    # ----------------------------------------------------------
    # MRI preprocessing
    # ----------------------------------------------------------    

    config = PreprocessingConfig(
        target_spacing=VoxelSpacing(1.0,1.0,1.0),
        target_shape=(64,64,64)
        )

    preprocessor = VolumePreprocessor(config)
    mri_processor = KneeMRIProcessor(preprocessor)

    # ----------------------------------------------------------
    # Build the model dataset
    # ----------------------------------------------------------

    model_sample_builder = ModelSampleBuilder(mri_processor)
    model_dataset = ModelDataset(samples, model_sample_builder)
    torch_dataset = PyTorchKneeDataset(model_dataset)

    # ----------------------------------------------------------
    # Create the Dataloader
    # ----------------------------------------------------------

    loader_factory = MOdelDataLoader(
        dataset=torch_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    loader = loader_factory.get_loader()

    assert len(loader) == 58
    print("Number of batches:", len(loader))

    # ----------------------------------------------------------
    # Load only the first batch
    # ----------------------------------------------------------

    sagittal, coronal, axial, target = next(iter(loader))

    assert isinstance(sagittal, torch.Tensor)
    assert isinstance(coronal, torch.Tensor)
    assert isinstance(axial, torch.Tensor)
    assert isinstance(target, torch.Tensor)

    assert tuple(sagittal.shape) == (1, 64, 64, 64)
    assert tuple(coronal.shape) == (1, 64, 64, 64)
    assert tuple(axial.shape) == (1, 64, 64, 64)
    assert tuple(target.shape) == (1, 12)

    print()
    print("Sagittal batch shape:", tuple(sagittal.shape))
    print("Coronal batch shape:", tuple(coronal.shape))
    print("Axial batch shape:", tuple(axial.shape))
    print("Target batch shape:", tuple(target.shape))
    print("Tension dtype::", sagittal.dtype)

    print()
    print("MODEL DATA LOADER INTEGRATION TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    main()