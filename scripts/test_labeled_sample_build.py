from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.preprocessing.series_selector import SeriesSelector

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

series_selector = SeriesSelector(train_series_dir=paths.train_series_dir)

train = metadata.train
labeled = train[train[LABEL_COLUMNS].notna().all(axis=1)]

print("=" * 70)
print("LABELED STUDY / SERIES ANALYSIS")
print("=" * 70)

print(f"Total studies: {len(train)}")
print(f"Labeled studies: {len(labeled)}")
print()

sagittal_count = 0
coronal_count = 0
axial_count = 0

for _, row in labeled.iterrows():

    study_uid = row["StudyInstanceUID"]

    series_rows = metadata.get_study_series(study_uid)

    selection = series_selector.select(study_uid=study_uid, series=series_rows)

    if selection.sagittal is not None:
        sagittal_count += 1

    if selection.coronal is not None:
        coronal_count += 1

    if selection.axial is not None:
        axial_count += 1

print(f"Labeled studies with sagittal: {sagittal_count}")
print(f"Labeled studies with coronal: {coronal_count}")
print(f"Labeled studies with axial: {axial_count}")

print()
print("=" * 70)


