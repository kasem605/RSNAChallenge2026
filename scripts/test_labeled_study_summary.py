from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader

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

train = metadata.train
labeled = train[train[LABEL_COLUMNS].notna().all(axis=1)]

print("=" * 70)
print("LABELED STUDY SUMMARY")
print("=" * 70)

print(f"Train studies: {len(train)}")
print(f"Labeled studies: {len(labeled)}")
print(f"Unlabeled studies: {len(train) - len(labeled)}")
print()

for uid in labeled["StudyInstanceUID"].head(10):
    print(uid)
    
print()
print("=" * 70)


