from pathlib import Path

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.preprocessing.series_selector import SeriesSelector

from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder
from ish_knee.dataset.labeled_dataset_builder import LabeledDatasetBuilder
from ish_knee.dataset.dataset_sample_validator import DatasetSampleValidator

DATASET_ROOT = r"D:\RSNA knee abnormality detection 2026"

print("=" * 70)
print("DATASET SAMPLE VALIDATOR TEST")
print("=" * 70)

paths = DatasetPaths.from_root(DATASET_ROOT)

metadata = MetadataReader(paths)

metadata.validate()

SeriesSelector = SeriesSelector(paths.train_series_dir)

sample_builder = DatasetSampleBuilder(metadata=metadata, series_selector=SeriesSelector)

labeled_builder = LabeledDatasetBuilder(sample_builder=sample_builder)

print()

print("Building labeled dataset sampless ...")

samples = labeled_builder.build_all()

print()
print("Total labeled samples:", len(samples))

validator = DatasetSampleValidator()

validator.validate(samples)

complete_count = validator.count_complete_samples(samples)

print()
print("Complete three-plane samples:", complete_count)

print()
print("Validation successful")
print()
print("DATASET SAMPLE VALIDATOR TEST PASSED")
print("=" * 70)

