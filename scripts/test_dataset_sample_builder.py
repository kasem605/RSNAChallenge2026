from pathlib import Path

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder

DATASET_ROOT = r"D:\RSNA knee abnormality detection 2026"

STUDY_UID = ("1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260")

paths = DatasetPaths.from_root(DATASET_ROOT)

metadata = MetadataReader(paths)

metadata.validate()

series_selector = SeriesSelector(paths.train_series_dir)

builder=DatasetSampleBuilder(metadata=metadata, series_selector=series_selector)

sample = builder.build(STUDY_UID)

print("=" * 70)
print("DATASET SAMPLE BUILDER TEST")
print("=" * 70)

print()
print("Study UID:")
print(sample.study_instance_uid)

print()
print("MRI planes:")
print("Sagittal:", sample.study.sagittal is not None)
print("Coronal:", sample.study.coronal is not None)
print("Axial:", sample.study.axial is not None)

print()
print("Plane count:")
print(sample.study.plane_count)

print()
print("Complete:")
print(sample.study.is_complete)

print()
print("Labels:")
print(sample.labels.as_tuple)

print()
print("Label count:")
print(sample.labels.count)

assert sample.study_instance_uid == STUDY_UID

assert sample.study.sagittal is not None
assert sample.study.coronal is not None
assert sample.study.axial is not None

assert sample.study.plane_count == 3
assert sample.study.is_complete

assert sample.labels.count == 12

print()
print("DATASET SAMPLE BUILDER TEST PASSED")
print("=" * 70)


