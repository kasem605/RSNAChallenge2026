
from pathlib import Path

import torch

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder
from ish_knee.dataset.labeled_dataset_builder import LabeledDatasetBuilder

from ish_knee.preprocessing.volume.volume_preprocessor import (
    VolumePreprocessor,
)
from ish_knee.preprocessing.volume.preprocessing_config import (
    PreprocessingConfig,
)
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing

from ish_knee.dataset.knee_mri_processor import KneeMRIProcessor
from ish_knee.model.model_sample_builder import ModelSampleBuilder
from ish_knee.model.knee_3d_cnn import Knee3DCNN


DATA_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")

TARGET_SHAPE = (128, 128, 128)


def main() -> None:
    print("Loading metadata...")

    paths = DatasetPaths.from_root(DATA_ROOT)
    metadata = MetadataReader(paths)
    metadata.validate()

    selector = SeriesSelector(paths.train_series_dir)
    sample_builder = DatasetSampleBuilder(metadata, selector)
    labeled_builder = LabeledDatasetBuilder(sample_builder)

    samples = labeled_builder.build_all()

    if not samples:
        raise ValueError("No fully labeled studies found.")

    # Test one real study without training or modifying checkpoints.
    sample = samples[0]

    config = PreprocessingConfig(
        target_spacing=VoxelSpacing(
            spacing_axis_0=1.0,
            spacing_axis_1=1.0,
            spacing_axis_2=1.0,
        ),
        target_shape=TARGET_SHAPE,
    )

    volume_preprocessor = VolumePreprocessor(config)
    mri_processor = KneeMRIProcessor(volume_preprocessor)
    model_sample_builder = ModelSampleBuilder(mri_processor)

    print(f"Study UID: {sample.study_instance_uid}")
    print("Preprocessing real MRI volumes...")

    # Adjust this call if your ModelSampleBuilder uses a different method.
    model_sample = model_sample_builder.build(sample)

    sagittal = torch.as_tensor(
        model_sample.input.sagittal, dtype=torch.float32
    ).unsqueeze(0)

    coronal = torch.as_tensor(
        model_sample.input.coronal, dtype=torch.float32
    ).unsqueeze(0)

    axial = torch.as_tensor(
        model_sample.input.axial, dtype=torch.float32
    ).unsqueeze(0)

    for name, volume in [
        ("Sagittal", sagittal),
        ("Coronal", coronal),
        ("Axial", axial),
    ]:
        print(f"{name} tensor shape: {tuple(volume.shape)}")
        assert tuple(volume.shape) == (1, *TARGET_SHAPE), (
            f"{name} has an unexpected shape."
        )
        assert torch.isfinite(volume).all(), (
            f"{name} contains NaN or infinite values."
        )

    model = Knee3DCNN().cpu().eval()

    print("Running inference...")

    with torch.inference_mode():
        logits = model(sagittal, coronal, axial)
        probabilities = torch.sigmoid(logits)

    assert logits.shape == (1, 12)
    assert torch.isfinite(probabilities).all()

    print(f"Logits shape: {tuple(logits.shape)}")
    print("Probabilities:")
    print(probabilities[0].tolist())
    print("REAL MRI INFERENCE TEST PASSED")


if __name__ == "__main__":
    main()
