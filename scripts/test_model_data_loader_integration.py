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

from ish_knee.model.knee_3d_cnn import Knee3DCNN

from ish_knee.model.model_sample_builder import ModelSampleBuilder
from ish_knee.model.model_dataset import ModelDataset
from ish_knee.model.pytorch_knee_dataset import PyTorchKneeDataset
from ish_knee.model.model_data_loader import ModelDataLoader

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
    # Audit label distribution across fully labeled studies
    # ----------------------------------------------------------
    import numpy as np

    label_names = [
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Medial OA",
        "Lateral OA",
        "PF OA",
        "Effusion",
        "Synovitis",
        "Baker's",
        "Contusion",
        "Fracture",
    ]

    label_matrix = np.asarray(
        [sample.labels.as_tuple for sample in samples],
        dtype=np.int32,
    )

    assert label_matrix.shape == (len(samples), len(label_names))
    assert np.isin(label_matrix, [0, 1]).all()

    print()
    print("LABEL DISTRIBUTION — FULLY LABELED STUDIES")
    print("-" * 65)

    for index, name in enumerate(label_names):
        positive = int(label_matrix[:, index].sum())
        negative = len(samples) - positive

        print(
            f"{name:20s} "
            f"Positive: {positive:2d}  "
            f"Negative: {negative:2d}"
        )

    print("-" * 65)
    print("Label distribution audit PASSED")

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

    loader_factory = ModelDataLoader(
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

    # ----------------------------------------------------------
    # Validate real MRI tensor values
    # ----------------------------------------------------------
    for name, volume in [
        ("Sagittal", sagittal),
        ("Coronal", coronal),
        ("Axial", axial),
    ]:
        assert torch.isfinite(volume).all().item(), (
            f"{name} volume contains NaN or infinity"
        )

        print(f"{name} min:", volume.min().item())
        print(f"{name} max:", volume.max().item())
        print(f"{name} mean:", volume.mean().item())
        print(f"{name} std:", volume.std().item())

        assert volume.std().item() > 0, (
            f"{name} volume is constant"
        )

    assert torch.isfinite(target).all().item()
    assert torch.all((target == 0) | (target == 1)).item()

    print("MRI tensor and label value checks PASSED")
    print()
    print("Sagittal batch shape:", tuple(sagittal.shape))
    print("Coronal batch shape:", tuple(coronal.shape))
    print("Axial batch shape:", tuple(axial.shape))
    print("Target batch shape:", tuple(target.shape))
    
    print("Tensor dtype:", sagittal.dtype)

    # ----------------------------------------------------------
    # Run the real MRI batch through the 3D CNN
    # ----------------------------------------------------------
    print()
    print("Testing Knee3DCNN with real MRI data...")

    model = Knee3DCNN()
    model.eval()

    with torch.no_grad():
        logits = model(sagittal, coronal, axial)

    assert isinstance(logits, torch.Tensor)
    assert tuple(logits.shape) == (1, 12)
    assert torch.isfinite(logits).all().item()

    print("Model output shape:", tuple(logits.shape))
    print("Model output dtype:", logits.dtype)
    print("All model outputs finite:", torch.isfinite(logits).all().item())

    print()
    print("MODEL DATA LOADER INTEGRATION TEST PASSED")
    print("REAL MRI CNN FORWARD-PASS TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    main()