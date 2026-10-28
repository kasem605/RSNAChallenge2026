
from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder

DATASET_ROOT = r"D:\RSNA knee abnormality detection 2026"

LABEL_COLUMNS = [
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
            "Fracture"
        ]

paths = DatasetPaths.from_root(DATASET_ROOT)

metadata = MetadataReader(paths)

series_selector = SeriesSelector(train_series_dir = paths.train_series_dir)

builder=DatasetSampleBuilder(metadata=metadata, series_selector=series_selector)

train = metadata.train

# ------------------------------------------------------------------
# Find one labeled study
# ------------------------------------------------------------------

labeled_rows = train[train[LABEL_COLUMNS].notna().all(axis=1)]

labeled_uid = labeled_rows.iloc[0]["StudyInstanceUID"]

# ------------------------------------------------------------------
# Find one unlabeled study
# ------------------------------------------------------------------

unlabeled_rows = train[train[LABEL_COLUMNS].isna().all(axis=1)]

unlabeled_uid = unlabeled_rows.iloc[0]["StudyInstanceUID"]

print("=" * 70)
print("DATASET SAMPLE BUILDER LABEL TEST")
print("=" * 70)

print()
print("Labeled study:")
print(labeled_uid)

print()
print("UnLabeled study:")
print(unlabeled_uid)

print()

# Make sure our test data is actually different
assert labeled_uid != unlabeled_uid, ("TEST ERROR: Labeled and unlabeled StudyInstanceUIDs are identical.")

# ------------------------------------------------------------------
# Test labeled study
# ------------------------------------------------------------------

print()
print("-" * 70)
print("TEST 1: LABELED STUDY")
print("-" * 70)

try:

    sample = builder.build(labeled_uid)

    print("DatasetSample created successfully")
    print()

    print("Study UID:")
    print(sample.study.study_instance_uid)

    print()
    print("Labels:")
    print(sample.labels.as_tuple)

    print()
    print("Label count:")
    print(sample.labels.count)

except Exception as exc:
    print("TEST FAILED")
    print(type(exc).__name__)
    print(str(exc))

# ------------------------------------------------------------------
# Test unlabeled study
# ------------------------------------------------------------------

print()
print("-" * 70)
print("TEST 2: UNLABELED STUDY")
print("-" * 70)

try:
    sample = builder.build(unlabeled_uid)

    print("ERROR:")
    print("Unlabeled study was incorrectly accepted.")

except ValueError as ve:
    print("Expected ValueError received")
    print()
    print(str(ve))

except Exception as exc:

    print("UNEXPECTED ERROR")
    print(type(exc).__name__)
    print(str(exc))


print()
print("=" * 70)
print("TEST PASSED COMPLETE")
print("=" * 70)


