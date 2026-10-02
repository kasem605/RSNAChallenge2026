from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder
from ish_knee.dataset.labeled_dataset_builder import LabeledDatasetBuilder


DATASET_ROOT = r"D:\RSNA knee abnormality detection 2026"

paths = DatasetPaths.from_root(DATASET_ROOT)
metadata = MetadataReader(paths)

series_selector = SeriesSelector(train_series_dir=paths.train_series_dir)

sample_builder = DatasetSampleBuilder(metadata=metadata, series_selector=series_selector)

dataset_builder = LabeledDatasetBuilder(sample_builder=sample_builder)

print("=" * 70)
print("LABELED DATASET BUILDER TEST")
print("=" * 70)

samples =  dataset_builder.build_all()

print()
print(f"DatasetSample count: {len(samples)}")

print()

if samples:

    first_sample = samples[0]

    print("First sample:")
    print()
    print("Study UID:")
    print(first_sample.study.study_instance_uid)

    print()
    print("labels:")
    print(first_sample.labels.as_tuple)

print()
print("=" * 70)

assert len(samples) == 58

print("LABELED DATASET BUILDER TEST")
print("=" * 70)