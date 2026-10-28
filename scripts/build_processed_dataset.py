
from pathlib import Path
import sys

# Make the src-based package importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.data.labeled_sample_builder import LabeledSampleBuilder
from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.preprocessing.volume.preprocessing_config import (
    PreprocessingConfig,
)
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
from ish_knee.preprocessing.volume.volume_preprocessor import (
    VolumePreprocessor,
)
from ish_knee.dataset.knee_mri_processor import KneeMRIProcessor
from ish_knee.dataset.processed_dataset_builder import (
    ProcessedDatasetBuilder,
)


def main() -> None:
    dataset_root = Path(r"D:\RSNA knee abnormality detection 2026")

    # Save to a new directory; existing output is never overwritten.
    output_dir = PROJECT_ROOT / "outputs" / "processed_dataset_v1"

    print("=" * 70)
    print("RSNA KNEE - PROCESSED DATASET BUILDER")
    print("=" * 70)

    # 1. Configure and validate the dataset.
    paths = DatasetPaths.from_root(dataset_root)
    paths.validate()
    paths.print_summary()

    # 2. Read metadata and build the fully labeled samples.
    metadata = MetadataReader(paths)
    metadata.validate()

    selector = SeriesSelector(
        train_series_dir=paths.train_series_dir
    )
    sample_builder = LabeledSampleBuilder(metadata, selector)
    samples = sample_builder.build()

    print(f"Fully labeled studies: {len(samples)}")

    if len(samples) != 58:
        raise ValueError(
            f"Expected 58 fully labeled studies, found {len(samples)}. "
            "Review the metadata before continuing."
        )

    # 3. Configure the existing 1 mm / 128^3 preprocessing pipeline.
    config = PreprocessingConfig(
        target_spacing=VoxelSpacing(1.0, 1.0, 1.0),
        target_shape=(128, 128, 128),
        normalize_intensity=True,
        padding_value=0.0,
    )

    preprocessor = VolumePreprocessor(config)
    processor = KneeMRIProcessor(preprocessor)

    # 4. Process each study and save its volumes and labels.
    builder = ProcessedDatasetBuilder(
        processor=processor,
        output_dir=output_dir,
        target_shape=(128, 128, 128),
    )

    builder.build(
        samples=samples,
        validation_fraction=0.30,
        seed=42,
    )


if __name__ == "__main__":
    main()
